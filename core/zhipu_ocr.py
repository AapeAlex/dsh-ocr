#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-ocr · 智谱 GLM-OCR 云端引擎（engine="glm"）

调智谱开放平台 GLM-OCR 模型（0.9B，OmniDocBench V1.5 SOTA 94.62）：
  · 输入：PDF ≤50MB/≤100 页；JPG/PNG 单图 ≤10MB
  · 输出：原生 Markdown（含表格/公式/图片链接），正好落进本插件
    「markdown 是工作格式」的知识库口径
  · 强项：复杂表格、公式、印章、代码文档、图文混排 —— 与扫描件财报
    （三线表）和盖章协议高度匹配

接口：chat/completions（OpenAI 兼容 v4）
  POST {BASE}/chat/completions   Authorization: Bearer <key>
  messages[].content = [{"type":"image_url","image_url":{"url":"data:<mime>;base64,..."}}]

密钥解析顺序：DSH_OCR_ZHIPU_KEY > ZHIPU_API_KEY > BIGMODEL_API_KEY
端点覆盖：DSH_OCR_ZHIPU_BASE_URL（默认 https://open.bigmodel.cn/api/paas/v4）

零第三方依赖：只用 urllib / base64 / json（stdlib）。
参考：docs.bigmodel.cn/cn/guide/models/vlm/glm-ocr
"""
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

DEFAULT_BASE = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-ocr"
PDF_MIME = "application/pdf"
IMG_MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class ZhipuOcrError(RuntimeError):
    """结构化错误（不抛 traceback，main 里转成 {ok:false}）"""


def resolve_key():
    for var in ("DSH_OCR_ZHIPU_KEY", "ZHIPU_API_KEY", "BIGMODEL_API_KEY"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    return None


def _data_uri(raw, mime):
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


def _post_chat(payload, api_key, base_url, timeout=180):
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise ZhipuOcrError("GLM-OCR HTTP %s: %s" % (e.code, body or e.reason))
    except urllib.error.URLError as e:
        raise ZhipuOcrError("GLM-OCR 网络错误: %s" % e.reason)
    except json.JSONDecodeError:
        raise ZhipuOcrError("GLM-OCR 返回非 JSON")


def glm_ocr_bytes(raw, mime, api_key=None, base_url=None, prompt=None):
    """对一段图片/PDF 字节跑 GLM-OCR，返回 (markdown_text, usage_dict)。"""
    key = api_key or resolve_key()
    if not key:
        raise ZhipuOcrError(
            "未配置智谱 API Key（设 ZHIPU_API_KEY 或 DSH_OCR_ZHIPU_KEY 后重试；"
            "或在 DSH 凭据服务里存智谱 Key 并导出到环境）")
    base = base_url or os.environ.get("DSH_OCR_ZHIPU_BASE_URL", DEFAULT_BASE)

    content = [{"type": "image_url", "image_url": {"url": _data_uri(raw, mime)}}]
    if prompt:
        content.append({"type": "text", "text": prompt})
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    data = _post_chat(payload, key, base)
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ZhipuOcrError("GLM-OCR 返回结构异常: %s" % json.dumps(data, ensure_ascii=False)[:300])
    return text or "", data.get("usage") or {}


def glm_ocr_file(path, api_key=None, base_url=None, prompt=None):
    """图片（jpg/png）→ 直接 base64；PDF → 原生整档解析，失败自动逐页渲染降级。"""
    ext = os.path.splitext(path)[1].lower()
    mime = IMG_MIMES.get(ext) or mimetypes.guess_type(path)[0]
    with open(path, "rb") as f:
        raw = f.read()
    size_mb = len(raw) / (1 << 20)

    if ext == ".pdf":
        # 原生 PDF：≤50MB / ≤100 页（GLM-OCR 文档口径）
        if size_mb > 50:
            raise ZhipuOcrError("PDF %.1fMB 超过 GLM-OCR 上限 50MB" % size_mb)
        try:
            return glm_ocr_bytes(raw, PDF_MIME, api_key, base_url, prompt)
        except ZhipuOcrError as e:
            # 某些网关不支持 data:application/pdf —— 降级为逐页渲染成 PNG 再调
            if "HTTP 4" not in str(e) and "结构异常" not in str(e):
                raise
            return _glm_ocr_pdf_by_pages(path, api_key, base_url, prompt)

    if mime not in ("image/png", "image/jpeg"):
        raise ZhipuOcrError("GLM-OCR 仅支持 JPG/PNG 图片（收到 %s）" % (mime or ext))
    if size_mb > 10:
        raise ZhipuOcrError("图片 %.1fMB 超过 GLM-OCR 上限 10MB" % size_mb)
    return glm_ocr_bytes(raw, mime, api_key, base_url, prompt)


def _glm_ocr_pdf_by_pages(path, api_key, base_url, prompt, dpi=150):
    """降级路径：PDF 逐页渲染 PNG → 逐页调 GLM-OCR，拼接为整档 Markdown。"""
    import io
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    parts, usage_total = [], {}
    try:
        for i in range(len(pdf)):
            img = pdf[i].render(scale=dpi / 72.0).to_pil().convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            text, usage = glm_ocr_bytes(buf.getvalue(), "image/jpeg", api_key, base_url, prompt)
            parts.append("## 第 %d 页\n\n%s" % (i + 1, text))
            if usage:
                usage_total["page_%d" % (i + 1)] = usage
    finally:
        pdf.close()
    return "\n\n".join(parts), usage_total


# ---------------------------------------------------------------------------
# Markdown 表格解析（table 模式 + glm 时保留「勾稽」纪律）
# ---------------------------------------------------------------------------
def parse_md_tables(md):
    """从 GLM-OCR 返回的 Markdown 里抽表格 → [[label, v1, v2, ...], ...]。

    只认标准 md 表（| a | b | 加 |---|---| 分隔行）；数值单元格做
    clean_number（千分位/括号负数），失败保留原文。
    """
    import re
    from ocr import clean_number
    rows = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s:\-|]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in _split_md_row(line)]
            header = [h for h in header if h != ""]
            rows.append(header)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in _split_md_row(lines[i])]
                cells = [c for c in cells if c != ""]
                parsed = []
                for j, c in enumerate(cells):
                    v = clean_number(c)
                    parsed.append(v if v is not None else c)
                rows.append(parsed)
                i += 1
            continue
        i += 1
    return rows


def _split_md_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.split("|")
