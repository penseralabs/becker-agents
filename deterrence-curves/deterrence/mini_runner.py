"""Minimal pytest-like runner, used only when pytest is not installed.

Discovers tests/**/test_*.py, runs every top-level test_* function, and prints
a pytest-style summary line. Install pytest for realistic behavior.
"""
import importlib.util
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    root = Path.cwd()
    sys.path.insert(0, str(root))
    passed = failed = errors = skipped = 0
    start = time.time()
    for f in sorted(root.glob("tests/**/test_*.py")):
        rel = f.relative_to(root).as_posix()
        modname = rel[:-3].replace("/", ".")
        try:
            spec = importlib.util.spec_from_file_location(modname, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
        except Exception:
            errors += 1
            print(f"ERROR collecting {rel}\n{traceback.format_exc(limit=3)}")
            continue
        for name, fn in list(vars(mod).items()):
            if not (name.startswith("test_") and callable(fn)):
                continue
            try:
                fn()
                passed += 1
            except Exception as e:
                if type(e).__name__ in ("Skipped", "SkipTest"):
                    skipped += 1
                else:
                    failed += 1
                    print(f"FAILED {rel}::{name} - {type(e).__name__}: {e}")
    parts = [f"{n} {k}" for n, k in ((failed, "failed"), (passed, "passed"),
                                     (skipped, "skipped"), (errors, "errors")) if n]
    print(f"{', '.join(parts) or 'no tests ran'} in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
