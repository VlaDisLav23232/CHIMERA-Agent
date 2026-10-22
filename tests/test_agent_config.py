import unittest
from dataclasses import replace
from pathlib import Path

from chimera_agent.config import ROOT, config_id, load_settings


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT / "configs/agent.json")

    def test_paths_are_absolute(self):
        self.assertTrue(self.settings.baseline_dir.is_absolute())

    def test_external_endpoints_rejected(self):
        for url in ("https://api.example.com/v1", "http://example.com/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                replace(self.settings, base_url=url)

    def test_bad_limits_rejected(self):
        for changes in ({"wall_seconds": 601}, {"top_p": 0}, {"max_tokens": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.settings, **changes)

    def test_config_identity_tracks_settings_and_code_not_paths(self):
        a = config_id(self.settings, {"source": "a"})
        self.assertEqual(a, config_id(replace(self.settings, data_dir=Path("/different")), {"source": "a"}))
        self.assertNotEqual(a, config_id(self.settings, {"source": "b"}))
        self.assertNotEqual(a, config_id(replace(self.settings, temperature=0), {"source": "a"}))
