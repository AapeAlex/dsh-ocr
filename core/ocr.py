#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-ocr · 可复用 OCR 引擎（扫描件 → 文本 / 表格 / 条款）

定位：一个可以被 DSH 插件（dsh-ocr）反复调用的 OCR 引擎。输入一份扫描 PDF 或图片，
按三种模式产出结构化结果：
  · text   整页 OCR：逐页文本 + 行数 + 平均置信度（最通用）。
  · table  财务报表：整页 OCR 定位数值列 → 列切条重 OCR → 按 y 归行 → 数字清洗
           → 轻量勾稽自校验（解决"无竖线三线表整页 OCR 漏识 ~1/3 数字"的问题）。
  · clause 合同条款：关键词正则抽取（回购/对赌/反稀释/优先清算…）+ 页码 + 引用。

引擎选择：RapidOCR（onnxruntime 跑 PaddleOCR PP-OCRv4 的 ONNX 模型）。
  它带 C 扩展（onnxruntime / cv2 / pypdfium2），在 DSH 内置 python 下会被 macOS
  hardened runtime 的 library validation 拒绝加载；必须用能加载这些 .so 的解释器
  （Anaconda，或任何设了 DSH_OCR_PYTHON 的解释器）。runner.js 负责选对解释器，
  本脚本只负责干活。

参考：若 RapidOCR 整页文本仍不够（复杂表格结构、多栏版式、表格单元格 span），
升级路径是 PaddleOCR 的文档解析流水线 paddleocr-doc-parsing（PP-Structure：版面
分析 + 表格结构识别）。本脚本的 table 模式是其"表格结构识别"的一版轻量实现
（x 聚类定列 + 列切条），详见 README「文档解析参考」。

用法：
  python ocr.py --input x.pdf --mode text --pages all --dpi 200
  python ocr.py --input x.pdf --mode table --pages 34,37,42 --dpi 400
  python ocr.py --input x.png --mode clause
产物：JSON 打到 stdout；--out-json / --out-md 落盘（可选）。
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# 数字清洗（逗号/句点混淆、千分位、括号负数）—— 移植自数据底表 FDD 抽取
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"[（(]?\s*[-\u2212]?\s*\d[\d,，.．]*\d?\s*[)）]?")


def clean_number(tok):
    """OCR 数字 → float。处理：全角/半角括号负号、千分位逗号、逗号句点混淆、尾部句点。

    OCR 常见混淆：千分位逗号被认成句点（52,163 → 52.163）、逗号句点双混
    （2,032.898 → 2.032.898）。按"千位分隔"规则归一，失败返回 None（不硬凑）。
    """
    t = (tok or "").strip().replace("（", "(").replace("）", ")").replace("，", ",").replace("．", ".")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
    if t.startswith(("-", "\u2212")):
        neg, t = True, t[1:].strip()
    t = t.replace(" ", "").replace(",", "")
    if t.endswith("."):
        t = t[:-1]
    if re.fullmatch(r"\d+\.\d{3}", t):            # 千分位逗号被认成句点
        t = t.replace(".", "")
    elif re.fullmatch(r"\d+\.\d{3},\d+", t):
        t = t.replace(".", "").replace(",", "")
    if t.count(".") > 1:
        head, _, tail = t.rpartition(".")
        t = head.replace(".", "") + "." + tail
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def numbers_in(text):
    return [v for v in (clean_number(m.group(0)) for m in NUM_RE.finditer(text or "")) if v is not None]


# ---------------------------------------------------------------------------
# 输入渲染（PDF → PIL 图 / 图片直读）
# ---------------------------------------------------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_pages(spec, total):
    """把 --pages 解析成 1-based 页号列表。支持 all / '1-10' / '3,7' / '1-3,9'。"""
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return list(range(1, total + 1))
    out, seen = [], set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            for p in range(min(a, b), max(a, b) + 1):
                if 1 <= p <= total and p not in seen:
                    seen.add(p)
                    out.append(p)
        else:
            p = int(part)
            if 1 <= p <= total and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def load_images(path, pages, dpi):
    """返回 [(page_no, PIL.Image, meta)]。PDF 用 pypdfium2 渲染，图片直读。"""
    ext = os.path.splitext(path)[1].lower()
    imgs = []
    if ext in IMG_EXTS:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        imgs.append((1, im, {"type": "image", "w": im.width, "h": im.height}))
        return imgs
    # PDF
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    total = len(pdf)
    if not pages:
        pages = list(range(1, total + 1))
    for pno in pages:
        img = pdf[pno - 1].render(scale=dpi / 72.0).to_pil().convert("RGB")
        imgs.append((pno, img, {"type": "pdf", "page": pno, "total": total}))
    pdf.close()
    return imgs


# ---------------------------------------------------------------------------
# OCR 引擎
# ---------------------------------------------------------------------------
def load_engine(lang):
    """懒加载 RapidOCR。失败给结构化错误 + 修复提示（而不是 traceback）。

    并发关键：RapidOCR/onnxruntime 默认 intra_op_num_threads=-1（用满 CPU 核数）。
    每次 doc_ocr 都 spawn 一个独立 Python 进程，N 个并发就是 N×核数 线程，CPU 严重
    超卖、单进程被拖到看起来「卡住」。这里把每进程的 onnx 线程数限制到 2（可用
    DSH_OCR_THREADS 覆盖，设 0 或负数表示不限制/走默认）。
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
        kwargs = {}
        if lang and lang.lower() in ("en", "eng"):
            kwargs["text_score"] = 0.5
        threads = os.environ.get("DSH_OCR_THREADS", "2").strip()
        try:
            n = int(threads)
        except ValueError:
            n = 2
        if n > 0:
            kwargs["intra_op_num_threads"] = n
            kwargs["inter_op_num_threads"] = n
        engine = RapidOCR(**kwargs)
        return engine
    except ImportError as e:
        raise RuntimeError(
            "OCR 引擎不可用：rapidocr_onnxruntime 未安装或 C 扩展无法加载（%s）。\n"
            "修复：用能加载 onnxruntime/cv2 的解释器运行（设 DSH_OCR_PYTHON 指向 Anaconda，"
            "或 `pip install rapidocr-onnxruntime` 到该解释器）。" % e)


def run_page(engine, img):
    """对一张 PIL 图跑 RapidOCR，返回行列表 [{x,y,w,h,text,conf}]（x/y 为左上，归一化到 0..1）。"""
    res, _ = engine(img)
    W, H = img.width, img.height
    lines = []
    if not res:
        return lines
    for r in res:
        box, text, score = r[0], r[1], r[2]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        lines.append({
            "x": round(x0 / W, 4), "y": round(y0 / H, 4),
            "w": round((x1 - x0) / W, 4), "h": round((y1 - y0) / H, 4),
            "text": str(text), "conf": round(float(score), 4),
        })
    return lines


def page_text(lines):
    return "\n".join(l["text"] for l in sorted(lines, key=lambda l: (l["y"], l["x"])))


# ---------------------------------------------------------------------------
# 模式：text / clause / table
# ---------------------------------------------------------------------------
CLAUSE_PATTERNS = {
    "回购": r"回购",
    "对赌/业绩承诺": r"对赌|业绩承诺|业绩补偿|估值调整",
    "反稀释": r"反稀释",
    "优先清算": r"优先清算|清算优先",
    "上市/退出": r"首次公开发行|上市|合格IPO",
    "股权转让限制": r"股权转让|优先购买权",
    "一票否决/特别表决": r"一票否决|特别表决|一致同意",
}


def clause_hits(pages_lines):
    """合同条款抽取：关键词 → 聚合（条款类型 + 命中页数 + 首页 + 最多 3 条去重引用）。"""
    by_type = {}
    for pno, lines in pages_lines:
        txt = "\n".join(l["text"] for l in lines)
        for name, pat in CLAUSE_PATTERNS.items():
            hits = by_type.setdefault(name, {"pages": set(), "quotes": []})
            for m in re.finditer(pat, txt):
                s = max(0, m.start() - 90)
                quote = re.sub(r"\s+", " ", txt[s:m.start() + 130]).strip()[:220]
                hits["pages"].add(pno)
                if len(hits["quotes"]) < 3 and quote not in hits["quotes"]:
                    hits["quotes"].append(quote)
    clauses = []
    for name, h in by_type.items():
        pages = sorted(h["pages"])
        clauses.append({
            "clause_type": name,
            "page_count_hit": len(pages),
            "first_page": pages[0] if pages else None,
            "quotes": h["quotes"],
        })
    clauses.sort(key=lambda c: c["clause_type"])
    return clauses


# --- table：数值列检测（1-D k-means） --------------------------------------
def kmeans1d(xs, k=3, iters=60):
    xs = sorted(xs)
    if len(xs) < k:
        k = max(1, len(xs))
    centers = [xs[int(len(xs) * (i + 0.5) / k)] for i in range(k)]
    for _ in range(iters):
        buckets = [[] for _ in range(k)]
        for x in xs:
            j = min(range(k), key=lambda t: abs(x - centers[t]))
            buckets[j].append(x)
        new = [sum(b) / len(b) if b else centers[j] for j, b in enumerate(buckets)]
        if all(abs(new[j] - centers[j]) < 1e-6 for j in range(k)):
            centers = new
            break
        centers = new
    return sorted(centers), [len(b) for b in buckets]


def detect_numeric_columns(lines, k=None, y_min=0.18, label_max=0.16, x_max=0.62):
    """用整页 OCR 的数字 token 实际 x 范围定位数值列。数字右对齐：right 是稳定锚点。

    返回 [(left, right, center)]，按 center 升序。排除长段中文（避免标签+数字合并）
    与过左的标签列、过右的说明列。
    """
    xs, extents = [], []
    for l in lines:
        if l["y"] < y_min or l["x"] > x_max:
            continue
        text = l["text"]
        if len(re.findall(r"[\u4e00-\u9fff]", text)) >= 10:
            continue
        for m in NUM_RE.finditer(text):
            if clean_number(m.group(0)) is None:
                continue
            cx = l["x"] + l["w"] * (m.start() / max(1, len(text)))
            if cx >= label_max:
                xs.append(cx)
                extents.append((l["x"], l["x"] + l["w"]))
    if len(xs) < 2:
        return []
    k = k if k and 0 < k <= 5 else 3
    centers, _ = kmeans1d(xs, k=k)
    cols = []
    for c in centers:
        lefts = [e[0] for x, e in zip(xs, extents) if abs(x - c) < 0.03]
        rights = [e[1] for x, e in zip(xs, extents) if abs(x - c) < 0.03]
        left = min(lefts) if lefts else max(0.0, c - 0.04)
        right = max(rights) if rights else min(1.0, c + 0.04)
        cols.append((round(left, 4), round(right, 4), round(c, 4)))
    return cols


def slice_and_ocr(img, engine, cols):
    """把整页按列边界切成「标签条 + N 个数值条」分别 OCR，避免相邻列合并。

    返回 [{label, col_values:[str]}] 按 y 归行；label 取标签条的文本，col_values 取各数值条。
    """
    from PIL import Image
    W, H = img.size
    bounds = [0.0] + [cols[i][0] for i in range(len(cols))] + [1.0]
    # 标签条 = 最左到第一个数值列左缘；数值条 = 各列左缘到下一列左缘
    strips = []
    # 标签条
    strips.append((0.0, bounds[1], "label"))
    for i in range(len(cols)):
        strips.append((cols[i][0], cols[i][1], "col%d" % i))
    ocr_by_strip = []
    for left, right, tag in strips:
        if right - left < 0.02:
            ocr_by_strip.append((tag, []))
            continue
        crop = img.crop((int(left * W), 0, int(right * W), H))
        lines = run_page(engine, crop)
        ocr_by_strip.append((tag, lines))
    return ocr_by_strip


def rebuild_rows(ocr_by_strip, col_count):
    """按 y 坐标把 N 个数值条的数字对回标签行。返回 [{label, cells:[value|None], conf}]。"""
    label_lines = next((ls for tag, ls in ocr_by_strip if tag == "label"), [])
    col_lines = [ls for tag, ls in ocr_by_strip if tag.startswith("col")][:col_count]
    rows = []
    for ll in sorted(label_lines, key=lambda l: (l["y"], l["x"])):
        label = ll["text"].strip()
        cells, confs = [], []
        for cl in col_lines:
            # 找与该标签行 y 最接近的数值
            best = None
            for c in cl:
                if abs(c["y"] - ll["y"]) < 0.02:
                    if best is None or abs(c["y"] - ll["y"]) < abs(best["y"] - ll["y"]):
                        best = c
            v = clean_number(best["text"]) if best else None
            cells.append(v)
            confs.append(best["conf"] if best else 0.0)
        rows.append({"label": label, "cells": cells,
                     "conf": round(sum(confs) / max(1, len(confs)), 3)})
    return rows


def balance_check(rows):
    """BS 平衡勾稽：资产总计 ≈ 负债合计 + 所有者权益合计（逐列，容差 1 元）。

    这是财务三表里最可泛化、最硬的一条自校验：三个总数是从 OCR 里独立读出来的，
    若两边对得上，说明列对齐与数字清洗整体可信；对不上则标出差异交给下游人工核对。
    用锚定精确匹配（^…$），避免「流动资产合计」被「资产合计」子串误配（实测踩过的坑）。
    """
    TOTAL_RE = {
        "assets": re.compile(r"^(总资产|资产总计|资产合计)$"),
        "liab": re.compile(r"^(负债总计|负债合计)$"),
        "equity": re.compile(r"^(所有者权益合计|股东权益合计|所有者权益总计|股东权益总计|所有者权益|净资产)$"),
    }

    def find_total(kind):
        for r in rows:
            label = re.sub(r"[\s·．.：:]+", "", r["label"])
            if TOTAL_RE[kind].match(label):
                return r
        return None

    assets = find_total("assets")
    liab = find_total("liab")
    equity = find_total("equity")
    if not assets or not liab or not equity:
        return []
    checks = []
    n = min(len(assets["cells"]), len(liab["cells"]), len(equity["cells"]))
    for ci in range(n):
        a, l, e = assets["cells"][ci], liab["cells"][ci], equity["cells"][ci]
        if a is not None and l is not None and e is not None:
            diff = round(abs(a - (l + e)), 2)
            checks.append({
                "col": ci,
                "assets_total": a,
                "liab_plus_equity": round(l + e, 2),
                "diff": diff,
                "balanced": diff <= 1,
            })
    return checks


def table_extract(engine, img):
    """table 模式主流程：整页 OCR 定列 → 列切条重 OCR → 归行 → 数字清洗 → 勾稽。"""
    whole = run_page(engine, img)
    cols = detect_numeric_columns(whole)
    if len(cols) < 2:
        return {"ok": False, "reason": "未检测到足够的数值列（可能非表格页）", "lines": whole}
    ocr_by_strip = slice_and_ocr(img, engine, cols)
    rows = rebuild_rows(ocr_by_strip, len(cols))
    checks = balance_check(rows)
    return {"ok": True, "columns": len(cols), "rows": rows,
            "balance_checks": checks, "lines": whole}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="dsh-ocr 引擎")
    ap.add_argument("--input", required=True)
    ap.add_argument("--pages", default="all")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--mode", default="text", choices=["text", "table", "clause"])
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--probe", action="store_true", help="只检测原生扩展能否加载（不跑 OCR）")
    args = ap.parse_args()

    # probe：给 runner/插件判断「这个解释器能不能跑 OCR」，不依赖输入文件
    if args.probe:
        probe = {"ok": True, "python": sys.executable, "imports": {}, "errors": []}
        for mod in ("onnxruntime", "cv2", "pypdfium2", "rapidocr_onnxruntime"):
            try:
                __import__(mod)
                probe["imports"][mod] = True
            except Exception as e:
                probe["imports"][mod] = False
                probe["errors"].append("%s: %s" % (mod, str(e).splitlines()[0][:200]))
        probe["ok"] = len(probe["errors"]) == 0
        print(json.dumps(probe, ensure_ascii=False))
        return

    if not os.path.exists(args.input):
        emit_error("input not found: %s" % args.input)
        return

    try:
        engine = load_engine(args.lang)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": "engine_unavailable",
                          "detail": str(e)}, ensure_ascii=False))
        return
    t0 = time.time()
    ext = os.path.splitext(args.input)[1].lower()

    # 先解析页数（PDF 需要先打开取总数）
    total_pages = 1
    if ext not in IMG_EXTS:
        try:
            import pypdfium2 as pdfium
            total_pages = len(pdfium.PdfDocument(args.input))
        except Exception as e:
            emit_error("PDF 打开失败: %s" % e)
            return
    pages = resolve_pages(args.pages, total_pages)

    images = load_images(args.input, pages, args.dpi)
    per_page = []
    all_text_lines = []
    for pno, img, meta in images:
        lines = run_page(engine, img)
        confs = [l["conf"] for l in lines]
        txt = page_text(lines)
        page_rec = {
            "page": pno,
            "chars": len(txt),
            "lines": len(lines),
            "avg_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "text": txt,
        }
        if args.mode == "table":
            te = table_extract(engine, img)
            page_rec["table"] = te
        per_page.append(page_rec)
        all_text_lines.append((pno, lines))

    clauses = clause_hits(all_text_lines) if args.mode == "clause" else []

    out = {
        "ok": True,
        "engine": "RapidOCR (PaddleOCR PP-OCRv4 ONNX)",
        "input": args.input,
        "mode": args.mode,
        "dpi": args.dpi,
        "pages_requested": pages,
        "page_count": len(images),
        "elapsed_s": round(time.time() - t0, 1),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pages": per_page,
        "clauses": clauses,
    }
    if args.out_json:
        write_json(args.out_json, out)
    if args.out_md:
        write_md(args.out_md, out)
    print(json.dumps(out, ensure_ascii=False))


def emit_error(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


def write_json(path, out):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def write_md(path, out):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    L = ["# OCR 结果 · %s\n" % os.path.basename(out["input"]),
         "- 引擎：%s" % out["engine"],
         "- 模式：%s · DPI %s · 页数 %s · 耗时 %ss\n" % (out["mode"], out["dpi"], out["page_count"], out["elapsed_s"])]
    for p in out["pages"]:
        L.append("\n## 第 %s 页（%s 行 · conf %s）\n" % (p["page"], p["lines"], p["avg_conf"]))
        if out["mode"] == "table" and p.get("table", {}).get("ok"):
            t = p["table"]
            L.append("| 项目 | %s |" % " | ".join("列%d" % i for i in range(t["columns"])))
            L.append("|---|" + "---|" * t["columns"])
            for r in t["rows"]:
                cells = ["%s" % (("" if v is None else format_number(v))) for v in r["cells"]]
                L.append("| %s | %s |" % (r["label"], " | ".join(cells)))
        else:
            L.append("```")
            L.append(p["text"])
            L.append("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def format_number(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(v))
    return "%.4f" % v


if __name__ == "__main__":
    main()
