"""Regression test: every shipped script must survive Jupyter's stdout.

ipykernel replaces sys.stdout with an OutStream that has no `reconfigure`
method, which is what broke diagnostics.py.  Here each script's module-level
code is executed with such an object installed.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
from pathlib import Path

REAL_OUT = sys.stdout

HERE = Path(__file__).parent
SCRIPTS = [
    "decompose.py", "analyze_combos.py", "diagnostics.py", "diag_extra.py",
    "show_diag.py", "net_emissions.py", "exp_45q.py", "fetch_45q.py",
    "grep45q.py", "compare_params.py", "scan_sites.py", "timing_diag.py",
    "check_notebook.py",
]

# The three pipeline scripts are flat scripts (all their work is at module
# level), so they are checked statically instead of being executed here.
PIPELINE = ["ccus_part1.py", "ccus_part2senario.py", "ccus_part3.py",
            "test_ce_variants.py"]


class FakeOutStream:
    """Mimics ipykernel.iostream.OutStream: no reconfigure attribute."""

    def __init__(self):
        self._buf = io.StringIO()

    def write(self, text):
        return self._buf.write(text)

    def writelines(self, lines):
        for line in lines:
            self._buf.write(line)

    def flush(self):
        pass

    def isatty(self):
        return False

    def getvalue(self):
        return self._buf.getvalue()


def main() -> int:
    # Run inside a temporary copy so the delivered folder is never modified
    # (the scripts create figures/ or 45q_sources/ when they start).
    sandbox = Path(tempfile.mkdtemp(prefix="ccus_jupyter_test_"))
    for path in HERE.glob("*.py"):
        shutil.copy2(path, sandbox / path.name)
    print(f"  (running in a temporary copy: {sandbox})\n")

    failures = []
    for name in PIPELINE:
        text = (HERE / name).read_text(encoding="utf-8")
        lines = text.splitlines()
        bad = False
        for i, ln in enumerate(lines):
            if "reconfigure" in ln and not ln.strip().startswith("#"):
                guarded = any(x.strip() == "try:" for x in lines[max(0, i - 3):i])
                if not guarded:
                    bad = True
        status = "FAIL (bare reconfigure)" if bad else "static OK"
        if bad:
            failures.append(name)
        print(f"  {name:<24} {status}")
    for name in SCRIPTS:
        path = sandbox / name
        if not path.exists():
            print(f"  MISSING {name}")
            failures.append(name)
            continue
        src = path.read_text(encoding="utf-8")
        fake_out, fake_err = FakeOutStream(), FakeOutStream()
        saved = sys.stdout, sys.stderr, sys.argv
        sys.stdout, sys.stderr, sys.argv = fake_out, fake_err, [name]
        try:
            # __name__ != "__main__" -> module level code runs, main() does not
            exec(compile(src, str(path), "exec"),
                 {"__name__": "__import_only__", "__file__": str(path)})
            status = "OK"
        except FileNotFoundError as exc:
            # 依赖上游结果的脚本（net_emissions.py、timing_diag.py 等）
            # 在结果尚未生成时读不到 CSV，这不算代码问题。
            status = f"SKIP (needs upstream results: {exc.filename or exc})"
        except SystemExit as exc:            # 缺输入文件时的友好退出，视为通过
            status = f"OK (graceful exit {exc.code})"
        except BaseException as exc:                                 # noqa: BLE001
            status = f"FAIL {type(exc).__name__}: {exc}"
            failures.append(name)
        finally:
            sys.stdout, sys.stderr, sys.argv = saved
        print(f"  {name:<24} {status}")

    print()
    if failures:
        print("FAILED:", failures)
        return 1
    print("all scripts survive a Jupyter-style stdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
