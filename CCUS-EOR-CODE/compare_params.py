"""Check that the three pipeline scripts really share the same parameters and
the same model class.

The three scripts (ccus_part1.py / ccus_part2senario.py / ccus_part3.py) each
carry a copy of `CCUSInvestmentModel`, so any parameter or formula edit must be
applied three times.  This script reads (does not execute) the three files,
extracts the parameter dictionaries and the class methods, and reports every
difference.

Run:  python compare_params.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).parent
SCRIPTS = ["ccus_part1.py", "ccus_part2senario.py", "ccus_part3.py"]


def brace_block(text: str, start: int) -> str:
    """Return the {...} block that starts at `start` (brace matching)."""
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    raise ValueError("unbalanced braces")


def extract_params(text: str) -> dict:
    """Evaluate the `params = {...}` literal (plus the constants it refers to)."""
    # ccus_part1/2 name it `params`, ccus_part3 names it `base_params`
    m = re.search(r"^[ \t]*(?:base_)?params\s*=\s*\{", text, flags=re.M)
    assert m, "parameter dictionary not found"
    literal = brace_block(text, text.index("{", m.start()))
    ns: dict = {"np": np}
    # 依次把字典里引用的模块级常量（如 I0_CNY）解析进来；
    # 若某个名字在文件里找不到，就用它的字面名字作为占位符，保证三个脚本仍可比较。
    for _ in range(20):
        try:
            return eval(literal, ns)  # noqa: S307
        except NameError as exc:
            name = str(exc).split("'")[1]
            mm = re.search(rf"^[ \t]*{name}\s*=\s*([^\n#]+)", text, flags=re.M)
            if not mm:
                ns[name] = f"<{name}>"
                continue
            try:
                ns[name] = eval(mm.group(1).strip(), ns)  # noqa: S307
            except Exception:                             # noqa: BLE001
                ns[name] = mm.group(1).strip()
    raise ValueError("could not evaluate the parameter dictionary")


def extract_methods(text: str) -> dict[str, str]:
    """Return {method_name: AST fingerprint} for the investment model class.

    Docstrings and comments are excluded, so only real code differences show up.
    """
    start = text.index("class CCUSInvestmentModel")
    body = text[start:]
    stop = re.search(r"\n(?=\S)", body)          # first column-0 line after the class
    if stop:
        body = body[: stop.start()]
    cls = ast.parse(body).body[0]
    out: dict[str, str] = {}
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        kept = [n for n in fn.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str))]
        clone = ast.FunctionDef(name=fn.name, args=fn.args,
                                body=kept or [ast.Pass()],
                                decorator_list=[], returns=None, type_comment=None)
        out[fn.name] = ast.dump(ast.fix_missing_locations(clone))
    return out


def main() -> None:
    texts = {name: (HERE / name).read_text(encoding="utf-8") for name in SCRIPTS}
    params = {name: extract_params(t) for name, t in texts.items()}
    methods = {name: extract_methods(t) for name, t in texts.items()}

    keys = sorted(set().union(*[set(p) for p in params.values()]))
    print("=" * 96)
    print("parameter comparison")
    print(f"{'parameter':<22}{'part1':<24}{'part2':<24}{'part3':<24}")
    n_diff = 0
    for k in keys:
        vals = [params[n].get(k, "<missing>") for n in SCRIPTS]
        flag = ""
        if not (vals[0] == vals[1] == vals[2]):
            flag = "   <<< DIFF"
            n_diff += 1
        print(f"{k:<22}{str(vals[0])[:22]:<24}{str(vals[1])[:22]:<24}"
              f"{str(vals[2])[:22]:<24}{flag}")
    print(f"\nparameters differing across scripts: {n_diff}")

    print("=" * 96)
    print("model-class method comparison (comments and whitespace ignored)")
    names = sorted(set().union(*[set(m) for m in methods.values()]))
    m_diff = 0
    for name in names:
        present = [name in methods[n] for n in SCRIPTS]
        bodies = [methods[n].get(name, "") for n in SCRIPTS]
        same = bodies[0] == bodies[1] == bodies[2]
        if not same or not all(present):
            m_diff += 1
            where = ",".join(n for n, p in zip(SCRIPTS, present) if not p)
            print(f"  DIFF  {name:<28}"
                  + (f" (missing in {where})" if where else " (code differs)"))
    print(f"\nclass methods differing across scripts: {m_diff}")
    if n_diff == 0 and m_diff == 0:
        print("\nOK: the three scripts agree on every parameter and every method.")
    else:
        print("\nNOT in sync: fix the differences above in all three files.")


if __name__ == "__main__":
    main()
