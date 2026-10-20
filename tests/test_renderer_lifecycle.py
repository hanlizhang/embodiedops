import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from renderer_lifecycle import dispose_in_owner_context


class RendererLifecycleTests(unittest.TestCase):
    def exercise(self, previous, fail=False):
        calls = []
        backend = SimpleNamespace(get_current_context=lambda: previous,
                                  make_context_current=lambda value: calls.append(("restore", value)))
        context = SimpleNamespace(gl_ctx=SimpleNamespace(
            _context=1, make_current=lambda: calls.append(("bind", 1))))

        def dispose(value):
            self.assertIs(value, context)
            calls.append(("free", 1))
            if fail:
                raise ValueError("free failed")

        if fail:
            with self.assertRaises(ValueError):
                dispose_in_owner_context(context, dispose, backend, lambda p: p)
        else:
            dispose_in_owner_context(context, dispose, backend, lambda p: p)
        return calls

    def test_dispose_old_renderer_restores_live_context(self):
        self.assertEqual(self.exercise(2), [("bind", 1), ("free", 1), ("restore", 2)])

    def test_does_not_restore_destroyed_owner(self):
        self.assertEqual(self.exercise(1), [("bind", 1), ("free", 1)])

    def test_restore_on_failure_and_no_context(self):
        self.assertEqual(self.exercise(None, True), [("bind", 1), ("free", 1), ("restore", None)])


if __name__ == "__main__":
    unittest.main()
