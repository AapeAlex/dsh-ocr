# dsh-ocr · OCR document recognition plugin

[English summary](#english-summary) · 中文说明

把 **OCR 做成可复用的 DSH 插件**：agent 工具 **`doc_ocr`** 把扫描件 PDF / 图片（无文本层）转成结构化文本。引擎为 **RapidOCR**（跑 PaddleOCR PP-OCRv4 的 ONNX 模型），全程本机运行，不发网络请求、不需要 API key。

## 一、三种模式

| mode | 用途 | 输出 |
|---|---|---|
| `text` | 整页 OCR（最通用） | 逐页文本 + 行数 + 平均置信度 |
| `table` | 财务报表（扫描件三线表） | 自动定位数值列 → 列切条重 OCR → 按 y 归行 → 数字清洗 → **资产负债表平衡勾稽** |
| `clause` | 合同条款 | 回购/对赌/反稀释/优先清算… 关键词命中 + 页码 + 引用片段 |

## 二、它解决什么问题

扫描件取数常见的四类问题，本插件都做了确定性处理：

1. **扫描件无文本层**：pypdf 抽不出文字，必须先渲染成图再识别；
2. **整页 OCR 漏识数字**：相邻列合并、数字右对齐截断，导致报表无法勾稽；
3. **逗号/句点混淆**：`52,163` 被认成 `52.163`，按千分位规则归一；
4. **OCR 数字不可全信**：只接受能被「资产总计 = 负债合计 + 所有者权益合计」勾住的数，勾不平就标出来交人工。

具体做法：

- `table` 模式用「数字 token 的 x 聚类」自动定位数值列，**列切条**分别 OCR（避免相邻列合并），数字右对齐以右缘为锚；
- `clean_number` 统一处理千分位 / 括号负数 / 逗号句点混淆；
- `balance_check` 做**锚定精确匹配**（`^资产总计$`），避免「流动资产合计」被「资产合计」子串误配；
- 条款抽取只返回「条款类型 + 页码 + 引用」，不照抄长段落。

## 三、运行环境

RapidOCR 依赖 `onnxruntime / opencv / pypdfium2` 等 **C 扩展 wheel**。在 macOS 上，如果 DSH 内置 Python 启用了 hardened runtime + Library Validation，可能拒绝加载这些 ad-hoc 签名的动态库：

```text
ImportError: dlopen(...): code signature ... different Team IDs
```

处理方式二选一：

1. 给内置 Python 注入 `com.apple.security.cs.disable-library-validation` entitlement 并 ad-hoc 重签（DSH 升级后可能需重跑）；
2. 用 `DSH_OCR_PYTHON` 指向一个能加载这些 wheel 的 Python（如 conda / Anaconda），插件会把 DSH 的 site-packages 追加到 `sys.path` 末尾，避免 NumPy ABI 冲突。

`runner.js` 的解释器选择顺序：

1. `DSH_OCR_PYTHON` 环境变量（显式指定，最优先）；
2. **DSH 内置 Python**；
3. `/opt/anaconda3/bin/python3`（兜底）。

对非 DSH 解释器，`runner.js` 把 DSH site-packages 经 `core/_boot/sitecustomize.py` **追加**到 `sys.path` 末尾——外部解释器自己的 numpy 优先，DSH 只补缺失包。

## 四、安装

从 GitHub：

```bash
dsh plugin add https://github.com/zirscard/dsh-ocr
```

从本地目录：

```bash
dsh plugin add link:/absolute/path/to/dsh-ocr
```

重启 DSH 后，新会话的工具列表出现 `doc_ocr`。

## 五、直接运行引擎（不经过 DSH）

```bash
PY=python3   # 换成能加载 RapidOCR 的解释器

# text：整页文本
"$PY" core/ocr.py --input scan.pdf --mode text --pages 34,37,42 --dpi 200

# table：财务报表（建议 dpi 300–400）
"$PY" core/ocr.py --input scan.pdf --mode table --pages 37 --dpi 300 \
  --out-json /tmp/p37.json --out-md /tmp/p37.md

# clause：合同条款
"$PY" core/ocr.py --input contract-page.png --mode clause
```

## 六、测试

```bash
python3 test/run_tests.py
```

覆盖：

1. **数字清洗**（纯函数）：千分位、括号负数、逗号句点混淆、页码解析；
2. **合成图 text 模式**：断言关键词命中 + 置信度；
3. **可选真实样本**：设置 `DSH_OCR_TEST_PDF`（扫描件 PDF）和/或 `DSH_OCR_TEST_IMAGE`（合同页图片）后，追加 table 勾稽与 clause 命中测试。

> 仓库不包含任何客户文档或扫描件。真实样本测试只在本地提供路径时运行。

## 七、并发控制

每次 `doc_ocr` 调用都会 spawn 一个独立 Python 进程并重新加载 RapidOCR 模型。并发调用时如果不限流，N 个进程 × onnxruntime 默认「满核线程」= CPU 严重超卖。本插件做了两层限制：

| 层级 | 机制 | 默认 | 覆盖 |
|---|---|---|---|
| 进程内 | `core/ocr.py` 设置 `intra_op_num_threads` / `inter_op_num_threads`（级联 Det/Cls/Rec） | **2** | `DSH_OCR_THREADS`（设 0/负数表示不限） |
| 进程间 | `runner.js` 模块级 FIFO 信号量，限制同时运行的 OCR 进程数 | **2** | `DSH_OCR_MAX_CONCURRENT` |

> 信号量是**同一 DSH host 进程内**的并发上限；跨 host 进程的多会话仍会各自 spawn 进程。需要跨进程限制时，设 `DSH_OCR_MAX_CONCURRENT=1` 并在外部串行调用。

## 八、复杂表格的升级路径

本插件的 `table` 模式是「表格结构识别」的**轻量实现**（x 聚类定列 + 列切条）。遇到复杂表格结构、多栏版式、单元格跨行跨列时，升级路径是 PaddleOCR 的文档解析流水线：

- [PP-Structure 文档分析](https://www.paddleocr.ai/latest/version2.x/ppstructure/overview.html)
- [PP-StructureV3 通用版面解析产线](https://paddlepaddle.github.io/PaddleX/3.0-rc/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html)

RapidOCR 本身就是 PaddleOCR PP-OCRv4 的 ONNX 运行时版（同检测/识别模型），所以升级是「换解析层」而非「换识别引擎」。

## 九、边界

- 不处理加密 PDF / 需要密码的文档（交给下游）；
- 不做定时任务、不做目录监控；
- 不编造数字：OCR 拿不到或勾稽不平的数，标出来交人工，不硬凑；
- 扫描件/图片才调 OCR；有文本层的 PDF 先用 pypdf 直读。

## English summary

`dsh-ocr` registers one `doc_ocr` tool for DeepSeek Harness. It turns scanned PDFs and images into structured text in three modes: `text` (per-page text and confidence), `table` (financial tables: auto-detected numeric columns, column-slice re-OCR, number cleaning, and a balance-sheet tie-out check), and `clause` (contract-clause keyword hits with page and quote). Everything runs locally through RapidOCR (PaddleOCR PP-OCRv4 ONNX models); no network request and no API key. The tool discovers an OCR-capable Python interpreter, limits concurrent OCR processes, and reports the exact cells that failed the tie-out instead of inventing numbers.

## License

MIT. RapidOCR / PaddleOCR models and libraries keep their own licenses.
