import json
import re

from chimera_agent_baseline.agent.form_fill import _extract_json_object
from chimera_agent_baseline.output.schema import Confidence, Weight, normalise_to_full_shape
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator

from chimera_agent.evidence import Claim, DecisionRecord, EvidenceItem, ToolCallRecord, message_text, reveal_sequence

# Same attempt budget as the upstream form-fill node, so parse robustness is not a difference between systems.
DECISION_ATTEMPTS = 3
MIN_FREE_TEXT = 40
# Gemma escapes apostrophes as \' inside JSON strings; JSON has no such escape. Only an unescaped backslash
# directly before an apostrophe is repaired, and every repair is counted in the trace.
APOSTROPHE_ESCAPE = re.compile(r"(?<!\\)((?:\\\\)*)\\'")

SYSTEM_PROMPT = (
    "You are completing the biopsy decision form for the prostate case you just analysed. "
    "You are given numbered evidence items: the case record shown up front [P], the clinical documents you "
    "retrieved [E…] and the guideline passages you retrieved [G…]. Every claim you write must be supported by "
    "the items it cites; do not state values or recommendations that no cited item contains. "
    "Output a SINGLE JSON object matching the requested shape, with no markdown fences and no other text."
)


def render_evidence(items: list[EvidenceItem]) -> str:
    blocks = []
    for item in items:
        if item.kind == "prompt":
            head = "Case record shown up front"
        elif item.kind == "clinical":
            head = f"{item.tool} ({item.section})"
        else:
            head = f"Guideline passage, page {item.page}, section {item.section}"
        body = item.content if isinstance(item.content, str) else json.dumps(item.content, ensure_ascii=False)
        blocks.append(f"[{item.id}] {head}:\n{body}")
    return "\n\n".join(blocks)


def decision_model(eligible: list[str]) -> type[BaseModel]:
    weights = create_model("DecisionWeights", __config__=ConfigDict(extra="forbid"),
                           **{name: (Weight, ...) for name in eligible})

    class DecisionForm(BaseModel):
        model_config = ConfigDict(extra="forbid")

        biopsy_decision: bool
        confidence: Confidence
        variable_weights: weights
        claims: list[Claim] = Field(min_length=2, max_length=4)

        @model_validator(mode="after")
        def long_enough(self):
            # The official reasoning field requires at least 40 characters; fail here so the model retries.
            if len(free_text(self.claims)) < MIN_FREE_TEXT:
                raise ValueError(f"claims together must be at least {MIN_FREE_TEXT} characters")
            return self

    return DecisionForm


def free_text(claims: list[Claim]) -> str:
    return " ".join(c.text.strip() for c in claims)


def decision_prompt(items: list[EvidenceItem], agent_message: str, eligible: list[str], guidance: str) -> str:
    weight_lines = ",\n".join(f'    "{v}": "not_used" | "noted" | "important" | "decisive"' for v in eligible)
    # agent_v1 passes no guidance and must get exactly the prompt of its dev run.
    guidance_block = f"{guidance}\n\n" if guidance else ""
    return (
        f"Evidence items:\n\n{render_evidence(items)}\n\n"
        f'Your own analysis (not evidence; do not cite it):\n"""\n{agent_message}\n"""\n\n'
        f"{guidance_block}"
        "Fill in the form. Weight each listed variable by how much it drove your recommendation. Write 2-4 "
        "claims naming the factors that drove it; each claim cites the IDs of the evidence items that support it.\n\n"
        "{\n"
        '  "biopsy_decision": true | false,\n'
        '  "confidence": "clear" | "borderline" | "uncertain",\n'
        f'  "variable_weights": {{\n{weight_lines}\n  }},\n'
        '  "claims": [{"text": "<one sentence>", "evidence": ["<ID>", ...]}, ...]\n'
        "}\n\n"
        f"`variable_weights` must contain exactly these keys: {', '.join(eligible)}. Only cite IDs listed above."
    )


def repair_apostrophes(text: str) -> tuple[str, int]:
    return APOSTROPHE_ESCAPE.subn(lambda match: match.group(1) + "'", text)


async def decide(model: BaseChatModel, items: list[EvidenceItem], agent_message: str, eligible: list[str],
                 guidance: str) -> tuple[DecisionRecord, int, int]:
    form = decision_model(eligible)
    prompt = decision_prompt(items, agent_message, eligible, guidance)
    errors = []
    for attempt in range(1, DECISION_ATTEMPTS + 1):
        hint = f"\n\nYour previous answer did not validate:\n{errors[-1]}\nOutput ONLY the corrected JSON." if errors else ""
        response = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt + hint)])
        try:
            text, repairs = repair_apostrophes(_extract_json_object(message_text(response.content)))
            parsed = form.model_validate_json(text)
        except ValidationError as error:
            errors.append(str(error))
            continue
        record = DecisionRecord(biopsy_decision=parsed.biopsy_decision, confidence=parsed.confidence,
                                variable_weights=parsed.variable_weights.model_dump(), claims=parsed.claims)
        return record, attempt, repairs
    raise RuntimeError(f"Decision form failed validation {DECISION_ATTEMPTS} times: {errors}")


def official_output(record: DecisionRecord, calls: list[ToolCallRecord]) -> tuple[str, dict]:
    weights = normalise_to_full_shape(1, {"variable_weights": {k: v.value for k, v in record.variable_weights.items()}})
    reasoning = {
        "confidence": record.confidence.value,
        "variable_weights": weights["variable_weights"],
        "reveal_sequence": reveal_sequence(calls),
        "free_text": free_text(record.claims),
    }
    return ("yes" if record.biopsy_decision else "no"), reasoning
