#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-ocr · 知识库文件预处理管线（7 阶段）

    discover → classify → extract → clean → structure → output → verify
                                              ↑                        ↓
                                           增量/缓存 ←── 失败人工队列

输入：单个文件或目录
输出：Obsidian-flavored markdown（YAML frontmatter）+ JSON + 附件

设计来源（对标 obsidian-import / foldmark / mineru-to-obsidian / text-extractor / vaultkeeper 等开源管线）：
  · obsidian-import：后端路由 + discover/extract/format/output 管线 + frontmatter 溯源
  · foldmark：OCR 懒触发（无文本层才 OCR）+ 单文件失败不中断
  · mineru-to-obsidian：按场景选提取器
  · text-extractor：提取结果缓存
  · vaultkeeper：附件先治理

用法：
  python pipeline.py --input 某目录 --output out/ --mode all
  python pipeline.py --input report.pdf --output out/ --mode all
  python pipeline.py --input 某目录 --output out/ --mode classify   # 只分类
产物：stdout 打印 JSON 摘要；--out-json / --out-md 落盘（可选）。
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr import (  # noqa: E402
    IMG_EXTS,
    clean_number,
    clause_hits,
    load_engine,
    page_text,
    run_page,
    table_extract,
)

EXCLUDE_DIRS = {".git", ".trash", "node_modules", ".ocr", "__pycache__", ".obsidian"}
EXCLUDE_FILES = {".DS_Store", "Thumbs.db"}
TEXT_EXTS = {".txt", ".md", ".markdown"}
DATA_EXTS = {".json", ".yaml", ".yml"}
OFFICE_EXTS = {".docx", ".xlsx", ".xlsm", ".pptx"}
PDF_TEXT_THRESHOLD = 100


# ---------------------------------------------------------------------------
# 阶段 0 · discover
# ---------------------------------------------------------------------------
def should_exclude(name):
    if name in EXCLUDE_FILES:
        return True
    if name.startswith("~$") or name.startswith(".~"):
        return True
    return False


def discover(input_path):
    """扫描文件或目录，返回文件清单。"""
    out = []
    if os.path.isfile(input_path):
        out.append(input_path)
    else:
        for root, dirs, fnames in os.walk(input_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fn in fnames:
                if should_exclude(fn):
                    continue
                out.append(os.path.join(root, fn))
    return sorted(out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 阶段 1 · classify
# ---------------------------------------------------------------------------
def pdf_text_chars(path):
    try:
        from pypdf import PdfReader
        r = PdfReader(path)
        if len(r.pages) == 0:
            return 0
        return len((r.pages[0].extract_text() or ""))
    except Exception:
        return 0


def classify(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "text_pdf" if pdf_text_chars(path) > PDF_TEXT_THRESHOLD else "scanned_pdf"
    if ext in IMG_EXTS:
        return "image"
    if ext == ".docx":
        return "docx"
    if ext in (".xlsx", ".xlsm"):
        return "xlsx"
    if ext == ".pptx":
        return "pptx"
    if ext == ".csv":
        return "csv"
    if ext in DATA_EXTS:
        return "data"
    if ext in TEXT_EXTS:
        return "text"
    return "unsupported"


# ---------------------------------------------------------------------------
# 阶段 2 · extract（后端路由）
# ---------------------------------------------------------------------------
def extract_text_pdf(path):
    from pypdf import PdfReader
    r = PdfReader(path)
    pages = []
    for i, p in enumerate(r.pages, 1):
        pages.append({"page": i, "text": p.extract_text() or ""})
    return {"kind": "text_pdf", "pages": pages, "page_count": len(pages),
            "backend": "pypdf", "confidence": None}


def extract_scanned_pdf(path, engine, dpi=200):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    pages = []
    confs = []
    for i in range(len(pdf)):
        img = pdf[i].render(scale=dpi / 72.0).to_pil().convert("RGB")
        lines = run_page(engine, img)
        txt = page_text(lines)
        pages.append({"page": i + 1, "text": txt})
        if lines:
            confs.append(sum(l["conf"] for l in lines) / len(lines))
    pdf.close()
    return {"kind": "scanned_pdf", "pages": pages, "page_count": len(pages),
            "backend": "rapidocr", "confidence": round(sum(confs) / len(confs), 3) if confs else None}


def extract_glm(path, kind):
    """engine=glm：智谱 GLM-OCR 云端解析（PDF 原生整档 / 图片 base64，输出原生 Markdown）。"""
    import zhipu_ocr
    text, usage = zhipu_ocr.glm_ocr_file(path)
    page_count = 1
    if kind == "scanned_pdf":
        try:
            import pypdfium2 as pdfium
            page_count = len(pdfium.PdfDocument(path))
        except Exception:
            page_count = 1
    return {"kind": kind, "pages": [{"page": 1, "text": text}], "page_count": page_count,
            "backend": "glm-ocr", "confidence": None, "usage": usage}


def extract_image(path, engine):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    lines = run_page(engine, img)
    txt = page_text(lines)
    conf = round(sum(l["conf"] for l in lines) / len(lines), 3) if lines else None
    return {"kind": "image", "pages": [{"page": 1, "text": txt}], "page_count": 1,
            "backend": "rapidocr", "confidence": conf}


def extract_docx(path):
    from docx import Document
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    tables = []
    for t in doc.tables:
        rows = []
        for row in t.rows:
            rows.append([c.text.strip() for c in row.cells])
        if rows:
            tables.append(rows)
    text = "\n\n".join(parts)
    if tables:
        text += "\n\n" + "\n\n".join(rows_to_md(t) for t in tables)
    return {"kind": "docx", "pages": [{"page": 1, "text": text}], "page_count": 1,
            "backend": "python-docx", "confidence": None, "tables": tables}


def extract_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sections = []
    tables = []
    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v) for v in row]
            if any(c.strip() for c in cells):
                rows.append(cells)
        if rows:
            sections.append("## %s\n\n%s" % (ws.title, rows_to_md(rows)))
            tables.append(rows)
    wb.close()
    return {"kind": "xlsx", "pages": [{"page": 1, "text": "\n\n".join(sections)}],
            "page_count": 1, "backend": "openpyxl", "confidence": None, "tables": tables}


def extract_pptx(path):
    from pptx import Presentation
    prs = Presentation(path)
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text)
        if texts:
            parts.append("## Slide %d\n\n%s" % (i, "\n\n".join(texts)))
    return {"kind": "pptx", "pages": [{"page": 1, "text": "\n\n".join(parts)}],
            "page_count": 1, "backend": "python-pptx", "confidence": None}


def extract_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.reader(f):
            rows.append([c.strip() for c in row])
    return {"kind": "csv", "pages": [{"page": 1, "text": rows_to_md(rows)}],
            "page_count": 1, "backend": "csv", "confidence": None, "tables": [rows]}


def extract_data(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    return {"kind": "data", "pages": [{"page": 1, "text": "```\n%s\n```" % raw}],
            "page_count": 1, "backend": "text", "confidence": None}


def extract_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    return {"kind": "text", "pages": [{"page": 1, "text": raw}], "page_count": 1,
            "backend": "text", "confidence": None}


def extract(path, kind, engine, dpi=200, ocr_backend="local"):
    if kind == "text_pdf":
        return extract_text_pdf(path)
    if kind == "scanned_pdf":
        if ocr_backend == "glm":
            return extract_glm(path, kind)
        return extract_scanned_pdf(path, engine, dpi)
    if kind == "image":
        if ocr_backend == "glm":
            return extract_glm(path, kind)
        return extract_image(path, engine)
    if kind == "docx":
        return extract_docx(path)
    if kind == "xlsx":
        return extract_xlsx(path)
    if kind == "pptx":
        return extract_pptx(path)
    if kind == "csv":
        return extract_csv(path)
    if kind == "data":
        return extract_data(path)
    if kind == "text":
        return extract_text(path)
    return {"kind": kind, "pages": [], "page_count": 0, "backend": None,
            "confidence": None, "unsupported": True}


# ---------------------------------------------------------------------------
# 阶段 3 · clean
# ---------------------------------------------------------------------------
def merge_wrapped_lines(text):
    """中文断行合并：上一行不以标点结尾、下一行以中文开头 → 拼回一行。"""
    lines = text.split("\n")
    out = []
    for ln in lines:
        if out and out[-1] and not re.search(r"[。！？；：、，）」』】\.\!\?\;:\,\"']$", out[-1]) \
                and re.match(r"^[\u4e00-\u9fff]", ln) and len(out[-1]) > 20:
            out[-1] = out[-1] + ln
        else:
            out.append(ln)
    return "\n".join(out)


def drop_headers_footers(pages):
    """页眉页脚：跨页高频重复行（出现次数 > max(2, 页数*0.5)）删除。"""
    if len(pages) < 3:
        return pages
    from collections import Counter
    cnt = Counter()
    for p in pages:
        for ln in set(l.strip() for l in p["text"].split("\n") if l.strip()):
            cnt[ln] += 1
    threshold = max(2, int(len(pages) * 0.5))
    noise = {ln for ln, c in cnt.items() if c >= threshold and len(ln) < 40}
    for p in pages:
        p["text"] = "\n".join(l for l in p["text"].split("\n") if l.strip() not in noise)
    return pages


def clean(extracted):
    pages = drop_headers_footers(extracted.get("pages", []))
    for p in pages:
        p["text"] = merge_wrapped_lines(p["text"])
    return extracted


# ---------------------------------------------------------------------------
# 阶段 4 · structure
# ---------------------------------------------------------------------------
def rows_to_md(rows):
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    head = "| " + " | ".join(norm[0]) + " |"
    sep = "|" + "---|" * width
    body = "\n".join("| " + " | ".join(r) + " |" for r in norm[1:])
    return head + "\n" + sep + ("\n" + body if body else "")


def structure(extracted, title):
    """标题层级：第X章/第X条 → markdown 标题；拼接页面。"""
    body_parts = []
    for p in extracted.get("pages", []):
        txt = p["text"]
        txt = re.sub(r"(?m)^第([一二三四五六七八九十百]+)章", r"## 第\1章", txt)
        txt = re.sub(r"(?m)^第([一二三四五六七八九十百]+)条", r"### 第\1条", txt)
        if extracted.get("page_count", 1) > 1:
            body_parts.append("## 第 %s 页\n\n%s" % (p["page"], txt))
        else:
            body_parts.append(txt)
    return body_parts


def frontmatter(rec):
    fm = {
        "source_path": rec["path"],
        "sha256": rec["sha256"],
        "kind": rec["kind"],
        "backend": rec["extracted"].get("backend"),
        "page_count": rec["extracted"].get("page_count"),
        "extracted_at": rec["extracted_at"],
    }
    if rec["extracted"].get("confidence") is not None:
        fm["ocr_confidence"] = rec["extracted"]["confidence"]
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, str):
            lines.append('%s: "%s"' % (k, v.replace('"', '\\"')))
        else:
            lines.append("%s: %s" % (k, v if v is not None else "null"))
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 阶段 5 · output
# ---------------------------------------------------------------------------
def output(rec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(rec["path"]))[0]
    md_path = os.path.join(out_dir, stem + ".md")
    json_path = os.path.join(out_dir, stem + ".json")

    fm = frontmatter(rec)
    body_parts = structure(rec["extracted"], stem)
    md = fm + "\n\n# " + stem + "\n\n" + "\n\n".join(body_parts)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": rec["path"],
            "sha256": rec["sha256"],
            "kind": rec["kind"],
            "backend": rec["extracted"].get("backend"),
            "page_count": rec["extracted"].get("page_count"),
            "confidence": rec["extracted"].get("confidence"),
            "tables": rec["extracted"].get("tables", []),
            "extracted_at": rec["extracted_at"],
        }, f, ensure_ascii=False, indent=1)
    return md_path, json_path


# ---------------------------------------------------------------------------
# 阶段 6 · verify
# ---------------------------------------------------------------------------
def verify(rec, md_path):
    problems = []
    try:
        body = open(md_path, encoding="utf-8").read()
    except Exception as e:
        return ["read_failed: %s" % e]
    if len(body.strip().split("\n")) < 3:
        problems.append("empty_body")
    if rec["extracted"].get("unsupported"):
        problems.append("unsupported_format")
    if rec["extracted"].get("page_count") == 0:
        problems.append("no_pages")
    return problems


# ---------------------------------------------------------------------------
# 阶段 7 · incremental（sha256 缓存）
# ---------------------------------------------------------------------------
def load_cache(out_dir):
    p = os.path.join(out_dir, "cache.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(out_dir, cache):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="dsh-ocr 预处理管线（7 阶段）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="all", choices=["all", "classify", "discover"])
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="忽略缓存，全部重跑")
    ap.add_argument("--engine", default="local", choices=["local", "glm"],
                    help="扫描件/图片的 OCR 后端：local=本机 RapidOCR（默认）；glm=智谱 GLM-OCR 云端")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = os.path.abspath(args.output)
    files = discover(args.input)

    result = {
        "ok": True,
        "mode": args.mode,
        "input": os.path.abspath(args.input),
        "output": out_dir,
        "discovered": len(files),
        "processed": [],
        "skipped": [],
        "failed": [],
        "elapsed_s": 0,
    }

    if args.mode == "discover":
        result["files"] = files
        result["elapsed_s"] = round(time.time() - t0, 1)
        print(json.dumps(result, ensure_ascii=False))
        return

    engine = None
    cache = load_cache(out_dir)

    for path in files[: args.max_files]:
        try:
            kind = classify(path)
            if args.mode == "classify":
                result["processed"].append({"path": path, "kind": kind})
                continue

            digest = sha256_file(path)
            if not args.force and cache.get(digest, {}).get("md") and os.path.exists(cache[digest]["md"]):
                result["skipped"].append(path)
                continue

            if engine is None and args.engine == "local":
                engine = load_engine("ch")

            rec = {"path": path, "sha256": digest, "kind": kind,
                   "extracted_at": datetime.now().isoformat(timespec="seconds")}
            rec["extracted"] = extract(path, kind, engine, args.dpi, ocr_backend=args.engine)
            clean(rec["extracted"])
            md_path, json_path = output(rec, out_dir)
            problems = verify(rec, md_path)
            cache[digest] = {"md": md_path, "json": json_path,
                             "processed_at": rec["extracted_at"], "kind": kind}
            result["processed"].append({
                "path": path, "kind": kind,
                "backend": rec["extracted"].get("backend"),
                "pages": rec["extracted"].get("page_count"),
                "md": md_path, "json": json_path,
                "problems": problems,
            })
        except Exception as e:
            result["failed"].append({"path": path, "error": str(e)[:300]})

    save_cache(out_dir, cache)
    result["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
