import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chimera_agent.evidence import (CASE_MISMATCH, Checks, Claim, DecisionRecord, Timings, Trace, build_ledger,
                                    reveal_sequence, run_checks)

PROMPT = "Case ID: c1. PSA: 7.2 ng/mL. PI-RADS score: 4."


def call(call_id, name, **args):
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def reply(call_id, name, payload, status="success"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ToolMessage(content=[{"type": "text", "text": text}], tool_call_id=call_id, name=name, status=status)


def synthetic_messages():
    return [
        HumanMessage(content=PROMPT),
        AIMessage(content="", tool_calls=[call("1", "get_mri_report", case_id="c1"),
                                          call("2", "get_family_history", case_id="c1")]),
        reply("1", "get_mri_report", {"case_id": "c1", "radiology_report": "Lesion 14 mm, volume 45 mL."}),
        reply("2", "get_family_history", {"case_id": "c1", "note": "No data available for this tool and case."}),
        AIMessage(content="", tool_calls=[call("3", "search_guidelines", query="PI-RADS 4 biopsy"),
                                          call("4", "get_psa_trend", case_id="other")]),
        reply("3", "search_guidelines", {"query": "q", "results": [
            {"text": "Biopsy for PI-RADS >= 3.", "page": 12, "section": "5.2", "score": 0.81},
            {"text": "Consider PSA density 0.15.", "page": 13, "section": "5.3", "score": 0.77}]}),
        reply("4", "get_psa_trend", f"{CASE_MISMATCH}: this consultation is case c1", status="error"),
        AIMessage(content="Recommend biopsy."),
    ]


def decision(*claims, weights=None):
    return DecisionRecord(biopsy_decision=True, confidence="clear",
                          variable_weights=weights or {"pirads": "decisive"}, claims=list(claims))


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.items, self.calls = build_ledger(PROMPT, synthetic_messages())

    def test_ids_kinds_and_order(self):
        self.assertEqual([i.id for i in self.items], ["P", "E1", "E2", "G1", "G2"])
        self.assertEqual([(c.order, c.tool, c.status, c.evidence_ids) for c in self.calls], [
            (1, "get_mri_report", "ok", ["E1"]), (2, "get_family_history", "no_data", ["E2"]),
            (3, "search_guidelines", "ok", ["G1", "G2"]), (4, "get_psa_trend", "rejected", [])])
        g1 = self.items[3]
        self.assertEqual((g1.page, g1.section, g1.score, g1.args), (12, "5.2", 0.81, {"query": "PI-RADS 4 biopsy"}))
        self.assertEqual(self.items[1].content, "Lesion 14 mm, volume 45 mL.")

    def test_reveal_sequence_skips_rejected_and_guidelines(self):
        self.assertEqual(reveal_sequence(self.calls), ["radiology_report", "family_history"])

    def test_unscored_guideline_hit_is_an_error_without_evidence(self):
        messages = [AIMessage(content="", tool_calls=[call("1", "search_guidelines", query="q")]),
                    reply("1", "search_guidelines", {"query": "q", "results": [
                        {"text": "Embedding service not running.", "page": None, "section": None, "score": None}]})]
        items, calls = build_ledger(PROMPT, messages)
        self.assertEqual([i.id for i in items], ["P"])
        self.assertEqual(calls[0].status, "error")
        missing_index = [AIMessage(content="", tool_calls=[call("1", "search_guidelines", query="q")]),
                         reply("1", "search_guidelines", {"query": "q", "results": [], "note": "Guidelines DB not available."})]
        self.assertEqual(build_ledger(PROMPT, missing_index)[1][0].status, "error")

    def test_unsupported_content_block_raises(self):
        messages = [AIMessage(content="", tool_calls=[call("1", "get_mri_report", case_id="c1")]),
                    ToolMessage(content=[{"type": "image", "url": "x"}], tool_call_id="1", name="get_mri_report")]
        with self.assertRaises(TypeError):
            build_ledger(PROMPT, messages)


class ChecksTest(unittest.TestCase):
    def setUp(self):
        self.items, self.calls = build_ledger(PROMPT, synthetic_messages())

    def test_supported_claims_pass(self):
        checks = run_checks(decision(Claim(text="PI-RADS 4 with a 14 mm lesion.", evidence=["P", "E1"]),
                                     Claim(text="Guidelines advise biopsy at PI-RADS 3 or above.", evidence=["G1"])),
                            self.items, self.calls)
        self.assertEqual(checks, Checks(missing_citations=[], unsupported_numbers=[], ungrounded_weights=[]))

    def test_missing_citation_and_unsupported_number_fail(self):
        checks = run_checks(decision(Claim(text="PSA 9.1 rising.", evidence=["E1"]),
                                     Claim(text="Volume 45 mL.", evidence=["E1", "E9"])),
                            self.items, self.calls)
        self.assertEqual([(i.claim, i.ids) for i in checks.missing_citations], [(1, ["E9"])])
        self.assertEqual([(i.claim, i.numbers) for i in checks.unsupported_numbers], [(0, ["9.1"])])

    def test_numbers_only_count_in_cited_evidence(self):
        checks = run_checks(decision(Claim(text="PSA 7.2.", evidence=["E1"])), self.items, self.calls)
        self.assertEqual([i.numbers for i in checks.unsupported_numbers], [["7.2"]])

    def test_weights_grounding(self):
        family = "get_family_history"
        ok = run_checks(decision(Claim(text="x", evidence=["P"]), weights={"fh": "noted"}), self.items, self.calls)
        self.assertEqual(ok.ungrounded_weights, [])
        calls = [c for c in self.calls if c.tool != family]
        bad = run_checks(decision(Claim(text="x", evidence=["P"]), weights={"fh": "noted", "psa": "not_used"}),
                         self.items, calls)
        self.assertEqual(bad.ungrounded_weights, ["fh"])


class TraceTest(unittest.TestCase):
    def test_round_trip_uses_schema_key(self):
        items, calls = build_ledger(PROMPT, synthetic_messages())
        d = decision(Claim(text="PI-RADS 4.", evidence=["P"]))
        trace = Trace(case_id="c1", evidence=items, tool_calls=calls, agent_message="Recommend biopsy.", decision=d,
                      checks=run_checks(d, items, calls), decision_attempts=1, llm_calls=4,
                      timings=Timings(agent_s=1.0, decision_s=0.5, total_s=1.6))
        data = json.loads(trace.model_dump_json(by_alias=True))
        self.assertEqual(data["schema"], "chimera_agent.trace/1")
        self.assertEqual(Trace.model_validate(data), trace)
        with self.assertRaises(ValueError):
            Trace.model_validate({**data, "schema": "chimera_agent.trace/0"})
