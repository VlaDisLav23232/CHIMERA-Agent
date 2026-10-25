import asyncio
import hashlib
import json
import unittest

from chimera_agent_baseline.output.schema import VARIABLES_BY_TASK, Task1Output
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from chimera_agent.decision import (SYSTEM_PROMPT, Calibration, decision_prompt, eau_policy, official_output,
                                    repair_escapes)
from chimera_agent.evidence import EvidenceItem, Gate, Trace, ToolCallRecord, grounded_variables
from chimera_agent.graph import PLAIN, Variant, run_case
from chimera_agent.prompts import AGENT_V2_DECISION_GUIDANCE

CONTEXT = "Case ID: c1\nPSA: 7.2 ng/mL\nPI-RADS score: 4"
ELIGIBLE = [v for v, tool in VARIABLES_BY_TASK[1].items() if tool is None]


class ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order, for the agent and the decision calls alike."""

    script: list
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def tool_call(call_id, name, **args):
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args, "type": "tool_call"}])


def form(**changes):
    value = {"biopsy_decision": True, "confidence": "clear",
             "variable_weights": {v: "not_used" for v in ELIGIBLE} | {"pirads": "decisive"},
             "claims": [{"text": "PI-RADS 4 lesion on the MRI report.", "evidence": ["P", "E1"]},
                        {"text": "The 14 mm lesion supports targeted biopsy.", "evidence": ["E1"]}]}
    return AIMessage(content=json.dumps(value | changes))


def make_tools(calls):
    async def get_mri_report(case_id: str) -> str:
        """MRI report."""
        calls.append(case_id)
        return json.dumps({"case_id": case_id, "radiology_report": "PI-RADS 4, 14 mm lesion."})

    async def get_psa_trend(case_id: str) -> str:
        """PSA trend."""
        calls.append(case_id)
        return json.dumps({"case_id": case_id, "psa_trend": [{"date": "2026-01-01", "val": 6.1}]})

    async def search_guidelines(query: str) -> str:
        """Guideline search."""
        calls.append(query)
        return json.dumps({"results": [{"text": "Biopsy for PI-RADS >= 4.", "page": 30, "section": "5.4",
                                        "score": 0.8}]})
    return [StructuredTool.from_function(coroutine=f) for f in (get_mri_report, get_psa_trend, search_guidelines)]


def run(script, calls=None, system="agent_v1", guidance="", variant=PLAIN):
    model = ScriptedModel(script=script, prompts=[])
    trace = asyncio.run(run_case(model, make_tools(calls if calls is not None else []), system, "system", guidance, "c1",
                                 CONTEXT, 25, variant))
    return trace, model


def variant(**changes):
    return Variant(**{"gate": None, "grounded_only": False, "mandatory_tools": (), "agent_tools": None,
                      "calibration": None} | changes)


CALIBRATION = Calibration(
    fitted_on="dev", cases={"yes": 3, "no": 2}, confidence={"yes": "clear", "no": "borderline"},
    variable_weights={d: {v: "noted" for v in VARIABLES_BY_TASK[1]} | {"pirads": "decisive"} for d in ("yes", "no")})


class GraphTest(unittest.TestCase):
    def test_loop_produces_trace_and_valid_official_output(self):
        trace, model = run([tool_call("1", "get_mri_report", case_id="c1"), AIMessage(content="Biopsy advised."),
                            form()])
        self.assertEqual([i.id for i in trace.evidence], ["P", "E1"])
        self.assertEqual(trace.agent_message, "Biopsy advised.")
        self.assertEqual((trace.decision_attempts, trace.llm_calls), (1, 3))
        self.assertEqual(trace.checks.missing_citations + trace.checks.unsupported_numbers, [])
        decision, reasoning = official_output(trace.decision, trace.tool_calls)
        self.assertEqual(decision, "yes")
        self.assertEqual(reasoning["reveal_sequence"], ["radiology_report"])
        self.assertEqual(set(reasoning["variable_weights"]), set(VARIABLES_BY_TASK[1]))
        self.assertNotIn("E1", reasoning["free_text"])
        Task1Output.model_validate({"case_id": "c1", "biopsy_decision": True, "confidence": reasoning["confidence"],
                                    "variable_weights": reasoning["variable_weights"],
                                    "reveal_sequence": reasoning["reveal_sequence"], "reasoning": reasoning["free_text"]})
        decision_prompt = model.prompts[-1][-1].content
        self.assertIn("[E1] get_mri_report (radiology_report)", decision_prompt)
        self.assertIn("Biopsy advised.", decision_prompt)

    def test_trace_records_system_and_guidance_reaches_decision(self):
        trace, model = run([AIMessage(content="Done."), form()], system="agent_v2", guidance=AGENT_V2_DECISION_GUIDANCE)
        self.assertEqual(trace.system, "agent_v2")
        self.assertIn(AGENT_V2_DECISION_GUIDANCE, model.prompts[-1][-1].content)

    def test_trace_system_is_required(self):
        trace, _ = run([AIMessage(content="Done."), form()])
        written = json.loads(trace.model_dump_json(by_alias=True))
        self.assertEqual((written["schema"], written["system"], written["gate"], written["calibrated"]),
                         ("chimera_agent.trace/3", "agent_v1", None, False))
        Trace.model_validate(written)
        del written["system"]
        with self.assertRaises(ValidationError):
            Trace.model_validate(written)

    def test_foreign_case_is_rejected_and_recorded(self):
        calls = []
        trace, _ = run([tool_call("1", "get_mri_report", case_id="other"), AIMessage(content="No data."),
                        form(claims=[{"text": "PI-RADS 4 on the case record shown.", "evidence": ["P"]},
                                     {"text": "PSA 7.2 ng/mL is above the usual cut-off.", "evidence": ["P"]}])],
                       calls)
        self.assertEqual(calls, [])
        self.assertEqual([(c.tool, c.status) for c in trace.tool_calls], [("get_mri_report", "rejected")])
        self.assertEqual([i.id for i in trace.evidence], ["P"])

    def test_invalid_form_is_retried(self):
        bad = AIMessage(content='{"biopsy_decision": true}')
        trace, model = run([AIMessage(content="Done."), bad, form(claims=[
            {"text": "PI-RADS 4 on the case record.", "evidence": ["P"]},
            {"text": "PSA 7.2 ng/mL supports biopsy.", "evidence": ["P"]}])])
        self.assertEqual(trace.decision_attempts, 2)
        self.assertIn("did not validate", model.prompts[-1][-1].content)

    def test_apostrophe_escapes_are_repaired_and_counted(self):
        backslash = chr(92)
        claims = [{"text": "PI-RADS 4 on the patient's record.", "evidence": ["P"]},
                  {"text": "PSA 7.2 ng/mL supports the patient's biopsy.", "evidence": ["P"]}]
        escaped = AIMessage(content=form(claims=claims).content.replace("'", backslash + "'"))
        trace, _ = run([AIMessage(content="Done."), escaped])
        self.assertEqual((trace.decision_attempts, trace.escape_repairs), (1, 2))
        self.assertEqual(trace.decision.claims[0].text, "PI-RADS 4 on the patient's record.")

    def test_repair_leaves_escaped_backslashes_alone(self):
        b = chr(92)
        self.assertEqual(repair_escapes("a" + b + "'b"), ("a'b", 1))
        self.assertEqual(repair_escapes("a" + b + b + "'b"), ("a" + b + b + "'b", 0))
        self.assertEqual(repair_escapes("a" + b + b + b + "'b"), ("a" + b + b + "'b", 1))

    def test_repair_drops_any_invalid_escape_and_keeps_valid_ones(self):
        b = chr(92)
        self.assertEqual(repair_escapes("5" + b + "% " + b + "_x"), ("5% _x", 2))
        valid = "".join(b + c for c in '"' + b + "/bfnrt") + b + "u00e9"
        self.assertEqual(repair_escapes(valid), (valid, 0))
        self.assertEqual(json.loads('"' + repair_escapes("PSA " + b + "> 10" + b + "n")[0] + '"'), "PSA > 10\n")

    def test_gate_fixes_the_decision_and_is_recorded(self):
        gate = eau_policy("4", 0.1)
        no = form(biopsy_decision=False)
        trace, model = run([AIMessage(content="Done."), no, form()], variant=variant(gate=gate))
        self.assertEqual((trace.decision.biopsy_decision, trace.decision_attempts), (True, 2))
        self.assertEqual(trace.gate, gate)
        self.assertIn('The decision is fixed: set "biopsy_decision" to true.', model.prompts[1][-1].content)

    def test_grounded_only_drops_variables_whose_section_was_not_revealed(self):
        weights = {v: "not_used" for v in ELIGIBLE if v not in ("bx", "dre")} | {"pirads": "decisive"}
        trace, model = run([tool_call("1", "get_mri_report", case_id="c1"), AIMessage(content="Done."),
                            form(variable_weights=weights)], variant=variant(grounded_only=True))
        self.assertNotIn('"bx"', model.prompts[-1][-1].content)
        _, reasoning = official_output(trace.decision, trace.tool_calls)
        self.assertEqual((reasoning["variable_weights"]["bx"], reasoning["variable_weights"]["dre"]),
                         ("not_used", "not_used"))

    def test_three_invalid_forms_fail(self):
        with self.assertRaisesRegex(RuntimeError, "failed validation 3 times"):
            run([AIMessage(content="Done.")] + [AIMessage(content="not json")] * 3)

    def test_weights_must_match_eligible_variables(self):
        extra = form(variable_weights={v: "not_used" for v in ELIGIBLE} | {"fh": "noted"})
        with self.assertRaises(RuntimeError):
            run([AIMessage(content="Done."), extra, extra, extra])


ITEMS = [EvidenceItem(id="P", kind="prompt", tool=None, args={}, order=0, content=CONTEXT),
         EvidenceItem(id="E1", kind="clinical", tool="get_mri_report", args={"case_id": "c1"}, order=1,
                      section="radiology_report", content="PI-RADS 4, 14 mm lesion."),
         EvidenceItem(id="G1", kind="guideline", tool="search_guidelines", args={"query": "q"}, order=2,
                      section="5.4", page=30, score=0.8, content="Biopsy is recommended for PI-RADS >= 4.")]
# SHA-256 of the agent_v1 decision messages for ITEMS, taken from the code that produced the agent_v1 dev run.
AGENT_V1_DECISION_SHA256 = "bd30038cd9b023fb55f8430579726bf9b1bc4349d4cc888882a570e56b1c6cff"


class DecisionPromptTest(unittest.TestCase):
    def test_agent_v1_decision_prompt_is_unchanged(self):
        text = SYSTEM_PROMPT + "\n\n" + decision_prompt(ITEMS, "Biopsy advised.", ELIGIBLE, "", None, False)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), AGENT_V1_DECISION_SHA256)

    def test_guidance_sits_between_analysis_and_form(self):
        prompt = decision_prompt(ITEMS, "Biopsy advised.", ELIGIBLE, AGENT_V2_DECISION_GUIDANCE, None, False)
        self.assertEqual(prompt.count(AGENT_V2_DECISION_GUIDANCE), 1)
        self.assertLess(prompt.index('Biopsy advised.\n"""'), prompt.index(AGENT_V2_DECISION_GUIDANCE))
        self.assertLess(prompt.index(AGENT_V2_DECISION_GUIDANCE), prompt.index("Fill in the form."))
        self.assertEqual(prompt.replace(AGENT_V2_DECISION_GUIDANCE + "\n\n", ""),
                         decision_prompt(ITEMS, "Biopsy advised.", ELIGIBLE, "", None, False))


class GateTest(unittest.TestCase):
    def test_eau_policy_by_pirads(self):
        self.assertEqual([eau_policy(v, 0.1).biopsy_decision for v in ("5", "4", "2", "1")], [True, True, False, False])
        self.assertEqual(eau_policy("4", 0.1).rule, "PI-RADS 4 is a positive MRI: biopsy is recommended")
        with self.assertRaises(ValueError):
            eau_policy("4a", 0.1)

    def test_eau_policy_decides_pirads_3_by_density(self):
        self.assertEqual(eau_policy("3", 0.25).biopsy_decision, True)
        self.assertEqual(eau_policy("3", 0.20).biopsy_decision, False)
        self.assertEqual(eau_policy("3", 0.11).rule, "PI-RADS 3 with PSA density 0.11 at or below 0.2: "
                                                     "biopsy can be omitted")
        self.assertEqual((eau_policy("5", 0.01).biopsy_decision, eau_policy("2", 0.9).biopsy_decision), (True, False))
        self.assertIsNone(eau_policy("NA", 0.3))
        with self.assertRaises(ValueError):
            eau_policy("3", "NA")

    def test_grounded_variables(self):
        all_vars = list(VARIABLES_BY_TASK[1])
        mri = [ToolCallRecord(order=1, tool="get_mri_report", args={}, status="ok", evidence_ids=["E1"])]
        self.assertEqual(set(grounded_variables(all_vars, [])), {"psa", "age", "comorbidity"})
        self.assertEqual(set(grounded_variables(all_vars, mri)),
                         {"psa", "age", "comorbidity", "pirads", "psad", "cspca", "vol"})


def claims_only(**changes):
    value = {"biopsy_decision": True,
             "claims": [{"text": "PI-RADS 4 lesion on the MRI report.", "evidence": ["P", "E1"]},
                        {"text": "The PSA trend is recorded.", "evidence": ["E2"]}]}
    return AIMessage(content=json.dumps(value | changes))


class AgentV3Test(unittest.TestCase):
    def v3(self, **changes):
        return variant(**{"gate": eau_policy("4", 0.1), "grounded_only": True,
                          "mandatory_tools": ("get_mri_report", "get_psa_trend"),
                          "agent_tools": ("search_guidelines",), "calibration": CALIBRATION} | changes)

    def test_documents_are_fetched_before_the_agent_and_only_offered_tools_run(self):
        calls = []
        trace, model = run([tool_call("1", "get_mri_report", case_id="c1"), AIMessage(content="Done."),
                            claims_only()], calls, system="agent_v3", variant=self.v3())
        self.assertEqual([(c.tool, c.status) for c in trace.tool_calls],
                         [("get_mri_report", "ok"), ("get_psa_trend", "ok"), ("get_mri_report", "error")])
        self.assertEqual(calls, ["c1", "c1"])
        first_agent_call = model.prompts[0]
        self.assertEqual(sum(type(m).__name__ == "ToolMessage" for m in first_agent_call), 2)
        _, reasoning = official_output(trace.decision, trace.tool_calls)
        self.assertEqual(reasoning["reveal_sequence"], ["radiology_report", "psa_trend"])

    def test_calibration_sets_confidence_and_grounded_weights(self):
        trace, model = run([AIMessage(content="Done."), claims_only()], system="agent_v3", variant=self.v3())
        self.assertTrue(trace.calibrated)
        self.assertEqual(trace.decision.confidence.value, "clear")
        _, reasoning = official_output(trace.decision, trace.tool_calls)
        weights = reasoning["variable_weights"]
        self.assertEqual((weights["pirads"], weights["psa"], weights["bx"], weights["dre"]),
                         ("decisive", "noted", "not_used", "not_used"))
        prompt = model.prompts[-1][-1].content
        self.assertNotIn('"variable_weights"', prompt)
        self.assertIn("Decision rule: PI-RADS 4 is a positive MRI", prompt)

    def test_calibrated_form_rejects_weights_from_the_model(self):
        with self.assertRaises(RuntimeError):
            run([AIMessage(content="Done.")] + [form()] * 3, system="agent_v3", variant=self.v3())

    def test_model_decides_when_no_rule_applies(self):
        trace, _ = run([AIMessage(content="Done."), claims_only(biopsy_decision=False)], system="agent_v3",
                       variant=self.v3(gate=None))
        self.assertEqual((trace.decision.biopsy_decision, trace.decision.confidence.value), (False, "borderline"))
