"""Packaging and documentation stay consistent with the code they describe."""
import os
import re
import unittest

import machaon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


class TestVersion(unittest.TestCase):
    def test_pyproject_citation_and_package_agree(self):
        pyproject = read("pyproject.toml")
        version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
        citation = read("CITATION.cff")
        cited = re.search(r"^version:\s*([^\s]+)", citation, re.M).group(1)
        self.assertEqual(version, machaon.__version__)
        self.assertEqual(cited, machaon.__version__)

    def test_changelog_has_an_entry_for_the_version(self):
        self.assertIn("## [%s]" % machaon.__version__, read("CHANGELOG.md"))

    def test_readme_and_install_docs_quote_the_real_test_count(self):
        suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"))
        count = suite.countTestCases()
        self.assertIn("tests-%d-" % count, read("README.md"))
        self.assertIn("Ran %d tests" % count, read("docs", "getting-started", "installation.md"))
        self.assertIn("Ran %d tests" % count, read("docs", "getting-started", "installation.zh.md"))


class TestPackaging(unittest.TestCase):
    def test_console_scripts_are_declared(self):
        pyproject = read("pyproject.toml")
        self.assertIn('machaon = "machaon.cli:main"', pyproject)
        self.assertIn('machaon-mcp = "machaon.mcp_server:main"', pyproject)

    def test_docs_extra_declares_the_i18n_plugin_used_by_mkdocs(self):
        pyproject = read("pyproject.toml")
        mkdocs = read("mkdocs.yml")
        self.assertIn("- i18n:", mkdocs)
        self.assertIn("mkdocs-static-i18n", pyproject)

    def test_core_dependency_is_still_numpy_only(self):
        pyproject = read("pyproject.toml")
        block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
        self.assertIn("numpy", block)
        self.assertNotIn("torch", block)


class TestPublicApi(unittest.TestCase):
    def test_every_export_resolves(self):
        for name in machaon.__all__:
            self.assertTrue(hasattr(machaon, name), name)

    def test_exported_functions_have_docstrings(self):
        for name in machaon.__all__:
            value = getattr(machaon, name)
            if callable(value) and not isinstance(value, type):
                self.assertTrue(getattr(value, "__doc__", None), name)


class TestDocumentationTone(unittest.TestCase):
    """The docs describe what the project does, not what it declines to do."""

    BANNED = (
        "out of scope",
        "not implemented",
        "not supported",
        "stops at GPT-2",
        "does not fit",
        "cannot edit",
    )

    def test_docs_do_not_use_self_limiting_phrases(self):
        paths = ["README.md", "CONTRIBUTING.md", "SECURITY.md"]
        for base, _, files in os.walk(os.path.join(ROOT, "docs")):
            paths.extend(
                os.path.relpath(os.path.join(base, name), ROOT)
                for name in files
                if name.endswith(".md")
            )
        for path in paths:
            text = read(path).lower()
            for phrase in self.BANNED:
                self.assertNotIn(phrase, text, "%s contains %r" % (path, phrase))

    def test_readme_links_to_live_docs(self):
        self.assertIn("https://s0ra-ai.github.io/Machaon/", read("README.md"))


class TestTranslations(unittest.TestCase):
    """Every page in the navigation exists in all four languages."""

    LANGUAGES = ("zh", "ja", "ko")

    def _nav_pages(self):
        nav = read("mkdocs.yml").split("\nnav:\n", 1)[1]
        return re.findall(r":\s*([A-Za-z0-9_./-]+\.md)\s*$", nav, re.M)

    def test_nav_covers_the_documentation_tree(self):
        pages = self._nav_pages()
        self.assertGreaterEqual(len(pages), 21)
        for page in pages:
            self.assertTrue(os.path.exists(os.path.join(ROOT, "docs", page)), page)

    def test_every_page_has_chinese_japanese_and_korean(self):
        for page in self._nav_pages():
            stem = page[:-3]
            for lang in self.LANGUAGES:
                path = os.path.join(ROOT, "docs", "%s.%s.md" % (stem, lang))
                self.assertTrue(os.path.exists(path), path)
                text = read("docs", "%s.%s.md" % (stem, lang))
                self.assertGreater(len(text), 200, path)
                self.assertTrue(text.startswith("# "), path)

    def test_language_switch_lists_all_four_locales(self):
        mkdocs = read("mkdocs.yml")
        for locale in ("en", "zh", "ja", "ko"):
            self.assertRegex(mkdocs, r"locale: %s\b" % locale)


if __name__ == "__main__":
    unittest.main()
