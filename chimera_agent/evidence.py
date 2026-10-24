import json
import re
from typing import Any, Literal

from chimera_agent_baseline.output.schema import Confidence, Weight, eligible_variables
from chimera_agent_baseline.tools.definitions import TASK1_TOOLS
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

TRACE_SCHEMA = "chimera_agent.trace/2"
GUIDELINE_TOOL = "search_guidelines"
CASE_MISMATCH = "case_id mismatch"
SECTION_BY_TOOL = {spec.name: spec.fields[0] for spec in TASK1_TOOLS}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Fixed scale legends of the baseline case template; their digits are not patient values.
PROMPT_LEGENDS = ("(1–2 low · 3 equivocal · 4–5 high risk)", "(clinically significant PCa = ISUP ≥ 2)")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(Strict):
    id: str
    kind: Literal["prompt", "clinical", "guideline"]
    tool: str | None
    args: dict[str, Any]
    order: int
    section: str | None = None
    page: int | str | None = None
    score: float | None = None
    content: Any


class ToolCallRecord(Strict):
    order: int
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "no_data", "rejected", "error"]
    evidence_ids: list[str]


class Claim(Strict):
    text: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class DecisionRecord(Strict):
    biopsy_decision: bool
    confidence: Confidence
    variable_weights: dict[str, Weight]
    claims: list[Claim]


class CitationIssue(Strict):
    claim: int
    ids: list[str]


class NumberIssue(Strict):
    claim: int
    numbers: list[str]


class Checks(Strict):
    missing_citations: list[CitationIssue]
    unsupported_numbers: list[NumberIssue]
    ungrounded_weights: list[str]


class Timings(Strict):
    agent_s: float
    decision_s: float
    total_s: float


class Trace(Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trace_schema: Literal["chimera_agent.trace/2"] = Field(TRACE_SCHEMA, alias="schema")
    case_id: str
    system: Literal["agent_v1"] = "agent_v1"
    evidence: list[EvidenceItem]
    tool_calls: list[ToolCallRecord]
    agent_message: str
    decision: DecisionRecord
    checks: Checks
    decision_attempts: int
    escape_repairs: int
    llm_calls: int
    timings: Timings


def message_text(content: str | list) -> str:
    # LangChain message content is a string or a list of content blocks; MCP replies use text blocks.
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        else:
            raise TypeError(f"Unsupported message content block: {block!r}")
    return "".join(parts)


def build_ledger(prompt: str, messages: list) -> tuple[list[EvidenceItem], list[ToolCallRecord]]:
    items = [EvidenceItem(id="P", kind="prompt", tool=None, args={}, order=0, content=prompt)]
    calls: list[ToolCallRecord] = []
    requested = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    clinical = guideline = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call = requested[message.tool_call_id]
        order = len(calls) + 1
        text = message_text(message.content)
        ids: list[str] = []
        if message.status == "error":
            # Tool node errors (the case guard, invalid arguments) are plain text, not tool JSON.
            status = "rejected" if text.startswith(CASE_MISMATCH) else "error"
        elif "error" in (payload := json.loads(text)):
            status = "error"
        elif call["name"] == GUIDELINE_TOOL:
            hits = payload["results"]
            # A note means the index is unavailable; a hit without a score is the "not running" placeholder.
            status = "ok" if "note" not in payload and all(h.get("score") is not None for h in hits) else "error"
            for hit in hits if status == "ok" else []:
                guideline += 1
                ids.append(f"G{guideline}")
                items.append(EvidenceItem(id=ids[-1], kind="guideline", tool=call["name"], args=call["args"],
                                          order=order, section=hit.get("section"), page=hit.get("page"),
                                          score=hit["score"], content=hit["text"]))
        else:
            section = SECTION_BY_TOOL[call["name"]]
            status = "ok" if section in payload else "no_data"
            clinical += 1
            ids.append(f"E{clinical}")
            items.append(EvidenceItem(id=ids[-1], kind="clinical", tool=call["name"], args=call["args"], order=order,
                                      section=section, content=payload.get(section, payload.get("note"))))
        calls.append(ToolCallRecord(order=order, tool=call["name"], args=call["args"], status=status,
                                    evidence_ids=ids))
    return items, calls


def revealed_tools(calls: list[ToolCallRecord]) -> set[str]:
    return {c.tool for c in calls if c.status in ("ok", "no_data") and c.tool in SECTION_BY_TOOL}


def reveal_sequence(calls: list[ToolCallRecord]) -> list[str]:
    sections: list[str] = []
    for call in calls:
        if call.status in ("ok", "no_data") and call.tool in SECTION_BY_TOOL:
            section = SECTION_BY_TOOL[call.tool]
            if section not in sections:
                sections.append(section)
    return sections


def _numbers(text: str) -> set[float]:
    return {float(n) for n in NUMBER.findall(text)}


def _evidence_text(item: EvidenceItem) -> str:
    if item.kind == "prompt":
        text = item.content
        for legend in PROMPT_LEGENDS:
            text = text.replace(legend, "")
        return text
    return json.dumps(item.content, ensure_ascii=False)


def run_checks(decision: DecisionRecord, items: list[EvidenceItem], calls: list[ToolCallRecord]) -> Checks:
    by_id = {item.id: item for item in items}
    missing, unsupported = [], []
    for index, claim in enumerate(decision.claims):
        absent = [i for i in claim.evidence if i not in by_id]
        if absent:
            missing.append(CitationIssue(claim=index, ids=absent))
        cited = " ".join(_evidence_text(by_id[i]) for i in claim.evidence if i in by_id)
        known = _numbers(cited)
        numbers = [n for n in NUMBER.findall(claim.text) if float(n) not in known]
        if numbers:
            unsupported.append(NumberIssue(claim=index, numbers=numbers))
    eligible = set(eligible_variables(1, revealed_tools(calls)))
    ungrounded = sorted(v for v, w in decision.variable_weights.items() if w != Weight.NOT_USED and v not in eligible)
    return Checks(missing_citations=missing, unsupported_numbers=unsupported, ungrounded_weights=ungrounded)
