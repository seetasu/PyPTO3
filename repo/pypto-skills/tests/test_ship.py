from __future__ import annotations

import unittest

from tests.skill_assertions import SKILLS

SKILL = SKILLS / "ship/SKILL.md"


class ShipSkillTests(unittest.TestCase):
    def test_ship_composes_publication_and_cleanup_contracts(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        self.assertIn("../auto-pr/SKILL.md", text)
        self.assertIn("../clean-branches/SKILL.md", text)
        self.assertIn("../../lib/repository/scope.md", text)

    def test_ship_confirms_merge_before_local_cleanup(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        merge_confirmation = text.find("require its server state to be merged")
        cleanup = text.find("## Return the local checkout")
        self.assertGreaterEqual(merge_confirmation, 0)
        self.assertGreater(cleanup, merge_confirmation)
        self.assertIn("only if its live OID still equals", text)
        self.assertIn("Do not enumerate or remove older `ship/*` branches", text)


if __name__ == "__main__":
    unittest.main()
