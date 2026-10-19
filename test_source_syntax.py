"""Check source syntax without importing GUI, audio or keyboard backends."""

import ast
from pathlib import Path
import unittest


class SourceSyntaxTests(unittest.TestCase):
    def test_project_python_sources_parse(self):
        for path in Path(__file__).parent.glob("*.py"):
            with self.subTest(source=path.name):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
