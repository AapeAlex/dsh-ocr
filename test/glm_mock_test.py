#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM-OCR 引擎离线端到端测试（本地 mock 服务器模拟智谱 chat/completions）。

不花一分钱验证：
  A. 请求构造：model=glm-ocr、image_url=data:image/png;base64、Bearer 头
  B. text 模式：返回的 Markdown 原样进入 pages[0].text
  C. table 模式：md 表格 → parse_md_tables → clean_number → BS 勾稽 平衡
  D. 无 Key：结构化错误 glm_unavailable（不 crash）

运行：
  python3 test/glm_mock_test.py
"""
import base64
import io
import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
PY = sys.executable

# 一张 GLM-OCR 风格的资产负债表 Markdown（含千分位、负数括号）
MOCK_MD = """# 资产负债表

| 项目 | 2024-12-31 | 2025-12-31 |
|---|---|---|
| 货币资金 | 52,163 | 273 |
| 应收账款 | 8,474 | 12,503 |
| 资产总计 | 608,209 | 486,041 |
| 负债合计 | 1,579,367 | 2,032,898 |
| 所有者权益合计 | (971,158) | (1,546,857) |
"""

captured = {}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        captured["auth"] = self.headers.get("Authorization", "")
        captured["path"] = self.path
        captured["body"] = body
        resp = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": MOCK_MD}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200},
        }, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # 造一张 1x1 测试 PNG
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    img_path = "/tmp/glm-mock-test.png"
    open(img_path, "wb").write(buf.getvalue())

    env = dict(os.environ,
               DSH_OCR_ZHIPU_KEY="test-key-123",
               DSH_OCR_ZHIPU_BASE_URL="http://127.0.0.1:%d" % port)

    ok, fail = [], []

    def check(name, cond, detail=""):
        (ok if cond else fail).append(name)
        print("  %s %s %s" % ("✅" if cond else "❌", name, detail))

    print("# A/B/C. engine=glm（mock 端点）")
    r = subprocess.run(
        [PY, os.path.join(PLUGIN, "core", "ocr.py"),
         "--input", img_path, "--mode", "table", "--engine", "glm"],
        capture_output=True, text=True, env=env, timeout=120)
    d = json.loads(r.stdout)

    check("请求路径 /chat/completions", captured.get("path") == "/chat/completions", captured.get("path", ""))
    check("Bearer 头", captured.get("auth") == "Bearer test-key-123")
    check("model=glm-ocr", captured.get("body", {}).get("model") == "glm-ocr")
    content = captured["body"]["messages"][0]["content"]
    iu = content[0]
    check("image_url 部件", iu.get("type") == "image_url")
    check("data:image/png;base64 前缀", iu["image_url"]["url"].startswith("data:image/png;base64,"))
    check("base64 与源文件一致", base64.b64decode(iu["image_url"]["url"].split(",", 1)[1]) == buf.getvalue())

    check("返回 ok", d.get("ok") is True)
    check("engine 标记", "GLM-OCR" in d.get("engine", ""))
    check("text 含表格标题", "资产负债表" in d["pages"][0]["text"])
    t = d["pages"][0]["table"]
    check("表格 ok", t.get("ok") is True)
    check("列数=2", t.get("columns") == 2, str(t.get("columns")))
    rows = {r0["label"]: r0["cells"] for r0 in t["rows"]}
    check("千分位清洗 52,163→52163", rows.get("货币资金", [None])[0] == 52163.0, str(rows.get("货币资金")))
    check("括号负数 (971,158)→-971158", rows.get("所有者权益合计", [None])[0] == -971158.0)
    bcs = t.get("balance_checks", [])
    check("勾稽 2/2 平衡（diff=0）", len(bcs) == 2 and all(b["balanced"] and b["diff"] == 0 for b in bcs),
          json.dumps(bcs, ensure_ascii=False))
    check("usage 透传", d.get("usage", {}).get("completion_tokens") == 200)

    print("\n# D. 无 Key 结构化错误")
    env2 = {k: v for k, v in os.environ.items() if "ZHIPU" not in k and "BIGMODEL" not in k}
    r2 = subprocess.run(
        [PY, os.path.join(PLUGIN, "core", "ocr.py"),
         "--input", img_path, "--mode", "text", "--engine", "glm"],
        capture_output=True, text=True, env=env2, timeout=60)
    d2 = json.loads(r2.stdout)
    check("glm_unavailable", d2.get("error") == "glm_unavailable", d2.get("detail", "")[:60])

    srv.shutdown()
    print("\n# 结果：%d 通过 / %d 失败" % (len(ok), len(fail)))
    if fail:
        print("失败项：", fail)
        sys.exit(1)
    print("全部通过 ✅（glm 全链路在 mock 端点下可用；真实端点只需 ZHIPU_API_KEY）")


if __name__ == "__main__":
    main()
