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

from chimera_agent.decision import SYSTEM_PROMPT, decision_prompt, official_output, repair_apostrophes
from chimera_agent.evidence import EvidenceItem, Trace
from chimera_agent.graph import run_case
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
    return [StructuredTool.from_function(coroutine=get_mri_report)]


def run(script, calls=None, system="agent_v1", guidance=""):
    model = ScriptedModel(script=script, prompts=[])
    trace = asyncio.run(run_case(model, make_tools(calls if calls is not None else []), system, "system", guidance, "c1",
                                 CONTEXT, 25))
    return trace, model


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
        self.assertEqual((written["schema"], written["system"]), ("chimera_agent.trace/2", "agent_v1"))
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
        self.assertEqual(repair_apostrophes("a" + b + "'b"), ("a'b", 1))
        self.assertEqual(repair_apostrophes("a" + b + b + "'b"), ("a" + b + b + "'b", 0))
        self.assertEqual(repair_apostrophes("a" + b + b + b + "'b"), ("a" + b + b + "'b", 1))

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
        text = SYSTEM_PROMPT + "\n\n" + decision_prompt(ITEMS, "Biopsy advised.", ELIGIBLE, "")
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), AGENT_V1_DECISION_SHA256)

    def test_guidance_sits_between_analysis_and_form(self):
        prompt = decision_prompt(ITEMS, "Biopsy advised.", ELIGIBLE, AGENT_V2_DECISION_GUIDANCE)
        self.assertEqual(prompt.count(AGENT_V2_DECISION_GUIDANCE), 1)
        self.assertLess(prompt.index('Biopsy advised.\n"""'), prompt.index(AGENT_V2_DECISION_GUIDANCE))
        self.assertLess(prompt.index(AGENT_V2_DECISION_GUIDANCE), prompt.index("Fill in the form."))
        self.assertEqual(prompt.replace(AGENT_V2_DECISION_GUIDANCE + "\n\n", ""),
                         decision_prompt(ITEMS, "Biopsy advised.", ELIGIBLE, ""))
