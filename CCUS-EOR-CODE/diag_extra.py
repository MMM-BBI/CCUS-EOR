"""Run only the two diagnostic blocks that were added after the first pass
(richer-basis analytic validation and the common-random-numbers resolution
test) and merge them into diagnostics_results.json."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import diagnostics as D  # noqa: E402  (import is safe: main() is guarded)

OUT = HERE / "diagnostics_results.json"

if __name__ == "__main__":
    M_e = 5000
    for _a in sys.argv[1:]:
        try:
            M_e = int(_a)
            break
        except ValueError:
            continue
    if os.environ.get("CCUS_QUICK_TEST"):
        M_e = 1000
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    res["C_analytic"] = D.run_analytic_validation()
    res["E_resolution"] = D.run_resolution(
        M_e, [42] if os.environ.get("CCUS_QUICK_TEST") else [7, 42, 2024])

    D.RESULTS = res
    D.make_figures()

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("merged into", OUT)
