import json
import re
from pathlib import Path
from typing import Literal

from chimera_agent_baseline.agent.form_fill import _extract_json_object
from chimera_agent_baseline.output.schema import VARIABLES_BY_TASK, Confidence, Weight, normalise_to_full_shape
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator

from chimera_agent.evidence import (Claim, DecisionRecord, EvidenceItem, Gate, ToolCallRecord, message_text,
                                    reveal_sequence)

# Same attempt budget as the upstream form-fill node, so parse robustness is not a difference between systems.
DECISION_ATTEMPTS = 3
MIN_FREE_TEXT = 40
# Gemma writes escapes JSON does not have, such as \' inside strings. An unescaped backslash before any character
# other than a valid JSON escape is dropped, and every repair is counted in the trace.
INVALID_ESCAPE = re.compile(r'(?<!\\)((?:\\\\)*)\\([^"\\/bfnrtu])')
# EAU: at PI-RADS 3 a PSA density at or below 0.20 ng/mL² allows the biopsy to be omitted.
PSAD_PIRADS3 = 0.20

SYSTEM_PROMPT = (
    "You are completing the biopsy decision form for the prostate case you just analysed. "
    "You are given numbered evidence items: the case record shown up front [P], the clinical documents you "
    "retrieved [E…] and the guideline passages you retrieved [G…]. Every claim you write must be supported by "
    "the items it cites; do not state values or recommendations that no cited item contains. "
    "Output a SINGLE JSON object matching the requested shape, with no markdown fences and no other text."
)


class Calibration(BaseModel):
    """Confidence and variable weights per decision, fitted on the dev references by evaluation/calibrate.py."""
    model_config = ConfigDict(extra="forbid")

    fitted_on: Literal["dev"]
    cases: dict[Literal["yes", "no"], int]
    confidence: dict[Literal["yes", "no"], Confidence]
    variable_weights: dict[Literal["yes", "no"], dict[str, Weight]]

    @model_validator(mode="after")
    def complete(self):
        for table in (self.cases, self.confidence, self.variable_weights):
            if set(table) != {"yes", "no"}:
                raise ValueError("calibration needs entries for both decisions")
        for weights in self.variable_weights.values():
            if set(weights) != set(VARIABLES_BY_TASK[1]):
                raise ValueError("calibration weights must cover exactly the Task 1 variables")
        return self


def load_calibration(path: Path) -> Calibration:
    return Calibration.model_validate_json(path.read_text())


def eau_policy(pirads: str, psad: float) -> Gate | None:
    # PI-RADS 4-5 is a positive MRI and 1-2 a negative one; at PI-RADS 3 the PSA density decides. NA (not reported)
    # has no rule and goes to the model.
    if pirads in ("4", "5"):
        return Gate(pirads=pirads, rule=f"PI-RADS {pirads} is a positive MRI: biopsy is recommended",
                    biopsy_decision=True)
    if pirads in ("1", "2"):
        return Gate(pirads=pirads, rule=f"PI-RADS {pirads} is a negative MRI: biopsy is not recommended",
                    biopsy_decision=False)
    if pirads == "NA":
        return None
    if pirads != "3":
        raise ValueError(f"Unexpected PI-RADS value {pirads!r}")
    if type(psad) not in (int, float):
        raise ValueError(f"PSA density must be a number, got {psad!r}")
    recommended = psad > PSAD_PIRADS3
    side, outcome = ("above", "biopsy is recommended") if recommended else ("at or below", "biopsy can be omitted")
    return Gate(pirads=pirads, rule=f"PI-RADS 3 with PSA density {psad} {side} {PSAD_PIRADS3}: {outcome}",
                biopsy_decision=recommended)


def gate_block(gate: Gate) -> str:
    value = "true" if gate.biopsy_decision else "false"
    return (f'Decision rule: {gate.rule}. The decision is fixed: set "biopsy_decision" to {value}. '
            "Write the claims for this decision from the evidence.\n\n")


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


def decision_model(eligible: list[str], gate: Gate | None, calibrated: bool) -> type[BaseModel]:
    weights = create_model("DecisionWeights", __config__=ConfigDict(extra="forbid"),
                           **{name: (Weight, ...) for name in eligible})
    decision_type = bool if gate is None else Literal[gate.biopsy_decision]
    # With a calibration the confidence and weights come from its table, so the model only decides and explains.
    rated = {} if calibrated else {"confidence": (Confidence, ...), "variable_weights": (weights, ...)}
    fields = create_model("DecisionFields", __config__=ConfigDict(extra="forbid"),
                          biopsy_decision=(decision_type, ...), **rated,
                          claims=(list[Claim], Field(min_length=2, max_length=4)))

    class DecisionForm(fields):
        @model_validator(mode="after")
        def long_enough(self):
            # The official reasoning field requires at least 40 characters; fail here so the model retries.
            if len(free_text(self.claims)) < MIN_FREE_TEXT:
                raise ValueError(f"claims together must be at least {MIN_FREE_TEXT} characters")
            return self

    return DecisionForm


def free_text(claims: list[Claim]) -> str:
    return " ".join(c.text.strip() for c in claims)


def decision_prompt(items: list[EvidenceItem], agent_message: str, eligible: list[str], guidance: str,
                    gate: Gate | None, calibrated: bool) -> str:
    weight_lines = ",\n".join(f'    "{v}": "not_used" | "noted" | "important" | "decisive"' for v in eligible)
    # agent_v1 passes no guidance and must get exactly the prompt of its dev run.
    guidance_block = f"{guidance}\n\n" if guidance else ""
    head = (
        f"Evidence items:\n\n{render_evidence(items)}\n\n"
        f'Your own analysis (not evidence; do not cite it):\n"""\n{agent_message}\n"""\n\n'
        f"{guidance_block}"
        f"{gate_block(gate) if gate else ''}"
    )
    if calibrated:
        return head + (
            "Fill in the form. Write 2-4 claims naming the factors that drove the recommendation; each claim cites "
            "the IDs of the evidence items that support it.\n\n"
            "{\n"
            '  "biopsy_decision": true | false,\n'
            '  "claims": [{"text": "<one sentence>", "evidence": ["<ID>", ...]}, ...]\n'
            "}\n\n"
            "Only cite IDs listed above."
        )
    return head + (
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


def repair_escapes(text: str) -> tuple[str, int]:
    return INVALID_ESCAPE.subn(lambda match: match.group(1) + match.group(2), text)


async def decide(model: BaseChatModel, items: list[EvidenceItem], agent_message: str, eligible: list[str],
                 guidance: str, gate: Gate | None, calibration: Calibration | None) -> tuple[DecisionRecord, int, int]:
    calibrated = calibration is not None
    form = decision_model(eligible, gate, calibrated)
    prompt = decision_prompt(items, agent_message, eligible, guidance, gate, calibrated)
    errors = []
    for attempt in range(1, DECISION_ATTEMPTS + 1):
        hint = f"\n\nYour previous answer did not validate:\n{errors[-1]}\nOutput ONLY the corrected JSON." if errors else ""
        response = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt + hint)])
        try:
            text, repairs = repair_escapes(_extract_json_object(message_text(response.content)))
            parsed = form.model_validate_json(text)
        except ValidationError as error:
            errors.append(str(error))
            continue
        if calibrated:
            key = "yes" if parsed.biopsy_decision else "no"
            confidence = calibration.confidence[key]
            weights = {v: calibration.variable_weights[key][v] for v in eligible}
        else:
            confidence, weights = parsed.confidence, parsed.variable_weights.model_dump()
        record = DecisionRecord(biopsy_decision=parsed.biopsy_decision, confidence=confidence,
                                variable_weights=weights, claims=parsed.claims)
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
