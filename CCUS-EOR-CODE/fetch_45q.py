"""Download and archive the authoritative sources for section 45Q.

Self-contained (standard library only).  The retrieved texts are written to
`45q_sources/` next to this script so that the parameter choices in the paper
can be audited offline:

    * IRS, Instructions for Form 8933 (12/2025)
    * 26 U.S. Code §45Q (Cornell LII, current text incl. Pub. L. 119-21)

Run:  python fetch_45q.py        (needs internet access)
"""

from __future__ import annotations

import html
import re
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = Path(__file__).parent / "45q_sources"
OUT.mkdir(exist_ok=True)

URLS = [
    ("irs_form8933_instructions_2025-12", "https://www.irs.gov/instructions/i8933"),
    ("usc_45Q_current", "https://www.law.cornell.edu/uscode/text/26/45Q"),
    ("irs_45q_credit_page",
     "https://www.irs.gov/credits-deductions/businesses/carbon-oxide-sequestration-credit"),
]

# 用于人工复核的关键表述
KEYS = ["12-year period", "applicable dollar amount", "multiplied by 5",
        "inflation", "per metric ton", "tertiary injectant", "utilization",
        "placed in service", "Parity"]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:      # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def strip(page: str) -> str:
    """Very small HTML-to-text converter (enough for .gov / law pages)."""
    page = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", page)
    page = re.sub(r"(?is)<br\s*/?>", "\n", page)
    page = re.sub(r"(?is)</(p|div|li|tr|h[1-6])>", "\n", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    page = html.unescape(page)
    return re.sub(r"[ \t\u00a0]+", " ", page).strip()


def main() -> None:
    for tag, url in URLS:
        try:
            text = strip(fetch(url))
        except Exception as exc:                                  # noqa: BLE001
            print(f"### {tag}: download failed ({exc})")
            continue
        path = OUT / f"{tag}.txt"
        path.write_text(text, encoding="utf-8")
        print("=" * 100)
        print(f"### {tag}  ({len(text)} chars)  {url}")
        print(f"    written to {path}")
        for key in KEYS:
            for m in list(re.finditer(re.escape(key), text, flags=re.I))[:1]:
                s = max(0, m.start() - 200)
                e = min(len(text), m.end() + 240)
                print(f"  --[{key}] ...{text[s:e]}...")


if __name__ == "__main__":
    main()
