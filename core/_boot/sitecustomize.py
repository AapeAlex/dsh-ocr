# sitecustomize.py —— 由 dsh-ocr/runner.js 通过 PYTHONPATH 注入
#
# 作用：把 DSH 的 site-packages **追加**到 sys.path 末尾。
# 为什么是追加而不是 PYTHONPATH 前置：
#   Anaconda 自带的 numpy/pandas 是按 NumPy 1.x 编译的；若把 DSH site-packages
#   前置，会优先载入 DSH 的 NumPy 2.3.5，导致 pyarrow/numexpr/bottleneck 报
#   "_ARRAY_API not found"（ABI 不兼容），pandas 启动刷屏。
#   追加到末尾 = Anaconda 自己的包优先（互相匹配），DSH 只补 Anaconda 缺失的包
#   （本插件需要的是 rapidocr_onnxruntime / onnxruntime / cv2 / pypdfium2 等
#   带 C 扩展、DSH 内置 python 因代码签名无法加载的包）。
import os
import sys

_sp = os.environ.get("V3_EXTRA_SITE")
if _sp and os.path.isdir(_sp) and _sp not in sys.path:
    if os.environ.get("V3_PREFER") == "extra":
        sys.path.insert(0, _sp)      # 备用：让 extras 优先（会引入 NumPy ABI 噪音）
    else:
        sys.path.append(_sp)
