#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-ocr self-test.

Usage:
    python3 test/run_tests.py

Optional real-sample tests (no fixtures are shipped with the repo):
    DSH_OCR_TEST_IMAGE=/path/to/contract-page.png python3 test/run_tests.py
    DSH_OCR_TEST_PDF=/path/to/scanned-report.pdf python3 test/run_tests.py

Coverage:
  A. Number cleaning (thousands separators, parenthesized negatives, comma/dot confusion)
  B. Synthetic-image text mode (keyword hit + confidence)
  C. Optional contract page, clause mode
  D. Optional scanned financial report, table mode with a balance-sheet tie-out check
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
CORE = os.path.join(PLUGIN, "core")
sys.path.insert(0, CORE)

import ocr  # noqa: E402  (the engine under test)

TEST_IMAGE = os.environ.get("DSH_OCR_TEST_IMAGE")
TEST_PDF = os.environ.get("DSH_OCR_TEST_PDF")
PDF_PAGE = int(os.environ.get("DSH_OCR_TEST_PDF_PAGE", "1"))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  %s %s %s" % ("✅" if cond else "❌", name, ("· " + detail) if detail else ""))


def main():
    print("# A. Number cleaning")
    check("thousands comma 52,163", ocr.clean_number("52,163") == 52163.0, str(ocr.clean_number("52,163")))
    check("dot as thousands separator 52.163", ocr.clean_number("52.163") == 52163.0, str(ocr.clean_number("52.163")))
    check("parenthesized negative (1,234)", ocr.clean_number("(1,234)") == -1234.0, str(ocr.clean_number("(1,234)")))
    check("decimal preserved 3.14", abs(ocr.clean_number("3.14") - 3.14) < 1e-9, str(ocr.clean_number("3.14")))
    check("page range 1-3,5", ocr.resolve_pages("1-3,5", 10) == [1, 2, 3, 5], str(ocr.resolve_pages("1-3,5", 10)))

    try:
        engine = ocr.load_engine("ch")
    except Exception as e:
        print("\n! RapidOCR is not importable in this interpreter: %s" % e)
        print("  Pure-function tests still ran. Set DSH_OCR_PYTHON to an OCR-capable interpreter for engine tests.")
        engine = None

    if engine is not None:
        print("\n# B. Synthetic image, text mode")
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (900, 160), "white")
            d = ImageDraw.Draw(img)
            font = None
            for fp in ("/System/Library/Fonts/STHeiti Medium.ttc",
                       "/System/Library/Fonts/Hiragino Sans GB.ttc",
                       "/System/Library/Fonts/PingFang.ttc"):
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, 40)
                        break
                    except Exception:
                        font = None
            if font is None:
                font = ImageFont.load_default()
            d.text((30, 40), "资产负债表 货币资金 52,163 千元", fill="black", font=font)
            lines = ocr.run_page(engine, img)
            txt = ocr.page_text(lines)
            conf = round(sum(l["conf"] for l in lines) / len(lines), 3) if lines else 0.0
            check("recognized lines", len(lines) > 0, "%d lines" % len(lines))
            check("keyword 货币资金", "货币资金" in txt, txt[:60])
            check("number 52,163 or 52163", "52,163" in txt or "52163" in txt, txt[:80])
            print("    conf=%s text=%r" % (conf, txt[:80]))
        except Exception as e:
            check("synthetic image test", False, "exception: %s" % e)

        print("\n# C. Optional contract page, clause mode")
        if TEST_IMAGE and os.path.exists(TEST_IMAGE):
            from PIL import Image
            img = Image.open(TEST_IMAGE).convert("RGB")
            lines = ocr.run_page(engine, img)
            clauses = ocr.clause_hits([(1, lines)])
            types = {c["clause_type"] for c in clauses}
            check("clause keyword hit", len(clauses) > 0, str(sorted(types)))
            print("    clause types=%s" % sorted(types))
        else:
            print("  (skipped: set DSH_OCR_TEST_IMAGE to a contract-page image)")

        print("\n# D. Optional scanned financial report, table mode")
        if TEST_PDF and os.path.exists(TEST_PDF):
            images = ocr.load_images(TEST_PDF, [PDF_PAGE], 300)
            img = images[0][1]
            te = ocr.table_extract(engine, img)
            check("numeric columns detected", te.get("columns") in (2, 3, 4), "columns=%s" % te.get("columns"))
            labels = [r["label"] for r in te.get("rows", [])]
            norm = [re.sub(r"[\s·．.：:]+", "", l) for l in labels]
            check("has 资产总计", any(l == "资产总计" for l in norm), "")
            check("has 负债合计", any(l == "负债合计" for l in norm), "")
            check("has 所有者权益合计", any(l == "所有者权益合计" for l in norm), "")
            checks = te.get("balance_checks", [])
            all_bal = len(checks) >= 1 and all(c["balanced"] for c in checks)
            check("balance-sheet tie-out passes", all_bal, json.dumps(checks, ensure_ascii=False)[:300])
            for c in checks:
                print("    col%d: assets %s vs liabilities+equity %s -> %s" % (
                    c["col"], c["assets_total"], c["liab_plus_equity"],
                    "OK" if c["balanced"] else "diff=" + str(c["diff"])))
        else:
            print("  (skipped: set DSH_OCR_TEST_PDF to a scanned financial report)")

    print("\n# Result: %d passed / %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("Failed:", FAIL)
        sys.exit(1)
    print("All passed ✅")


if __name__ == "__main__":
    main()
