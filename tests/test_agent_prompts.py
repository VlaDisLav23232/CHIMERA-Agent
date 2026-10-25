import re
import unittest

from chimera_agent_baseline.agent.prompts import build_system_prompt
from chimera_agent_baseline.tools.definitions import TASK1_TOOLS

from chimera_agent.prompts import AGENT_V2_DECISION_GUIDANCE, AGENT_V2_SYSTEM_PROMPT
from chimera_agent.worker import prompts_for

PROMPTS = (AGENT_V2_SYSTEM_PROMPT, AGENT_V2_DECISION_GUIDANCE)


class AgentV2PromptTest(unittest.TestCase):
    def test_system_prompt_offers_exactly_the_task1_tools(self):
        offered = set(re.findall(r"`(\w+)`", AGENT_V2_SYSTEM_PROMPT))
        self.assertEqual(offered, {spec.name for spec in TASK1_TOOLS} | {"search_guidelines"})

    def test_no_misleading_baseline_hints(self):
        for prompt in PROMPTS:
            self.assertNotIn("get_pathology_report", prompt)
            self.assertNotIn("duplicates the PSA trend", prompt)

    def test_no_case_identifiers(self):
        # Prompt text is general knowledge; nothing taken from a particular case.
        for prompt in PROMPTS:
            self.assertNotIn("PT-pseudo_", prompt)

    def test_guidance_defines_every_confidence_level(self):
        for level in ("clear", "borderline", "uncertain"):
            self.assertIn(f'"{level}":', AGENT_V2_DECISION_GUIDANCE)


class PromptsForTest(unittest.TestCase):
    def test_agent_v1_keeps_the_baseline_prompt_and_no_guidance(self):
        self.assertEqual(prompts_for("agent_v1"), (build_system_prompt(), ""))

    def test_agent_v2_gets_our_prompts(self):
        self.assertEqual(prompts_for("agent_v2"), (AGENT_V2_SYSTEM_PROMPT, AGENT_V2_DECISION_GUIDANCE))

    def test_agent_v3_prompt_offers_only_guideline_search(self):
        system_prompt, guidance = prompts_for("agent_v3")
        self.assertEqual((set(re.findall(r"`(\w+)`", system_prompt)), guidance), ({"search_guidelines"}, ""))
        self.assertNotIn("PT-pseudo_", system_prompt)


if __name__ == "__main__":
    unittest.main()
