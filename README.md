# dsh-ocr · OCR 文档识别 + 知识库预处理插件

[English summary](#english-summary) · 中文说明

把 **OCR 与批量文档预处理做成可复用的 DSH 插件**：

- 工具 **`doc_ocr`**：把扫描件 PDF / 图片（无文本层）转成结构化文本，**双引擎**——`local` = RapidOCR（PaddleOCR PP-OCRv4 ONNX，本机运行）/ `glm` = 智谱 GLM-OCR 云端（原生 Markdown，含表格与公式）。
- 工具 **`doc_preprocess`**：7 阶段管线，把一个目录/一批原始文件（PDF/Office/图片/文本）整理成 Obsidian 可读知识库（markdown + JSON + sha256 增量缓存）。

## 一、`doc_ocr`：三种模式 × 双引擎

| mode | 用途 | 输出 |
|---|---|---|
| `text` | 整页 OCR（最通用） | 逐页文本 + 行数 + 平均置信度 |
| `table` | 财务报表（扫描件三线表） | 自动定位数值列 → 列切条重 OCR → 按 y 归行 → 数字清洗 → **资产负债表平衡勾稽** |
| `clause` | 合同条款 | 回购/对赌/反稀释/优先清算… 关键词命中 + 页码 + 引用片段 |

| engine | 说明 |
|---|---|
| `local`（默认） | RapidOCR，跑 PaddleOCR PP-OCRv4 的 ONNX 模型，本机推理，不发网络请求 |
| `glm` | 智谱 GLM-OCR 云端（0.9B，OmniDocBench SOTA），输出原生 Markdown；复杂表格 / 公式 / 盖章 / 图文混排更强，需要 API key |

### 它解决什么问题

1. **扫描件无文本层**：pypdf 抽不出文字，必须先渲染成图再识别；
2. **整页 OCR 漏识数字**：相邻列合并、数字右对齐截断，导致报表无法勾稽；
3. **逗号/句点混淆**：`52,163` 被认成 `52.163`，按千分位规则归一；
4. **OCR 数字不可全信**：只接受能被「资产总计 = 负债合计 + 所有者权益合计」勾住的数，勾不平就标出来交人工。

具体做法：

- `table` 模式用「数字 token 的 x 聚类」自动定位数值列，**列切条**分别 OCR（避免相邻列合并），数字右对齐以右缘为锚；
- `clean_number` 统一处理千分位 / 括号负数 / 逗号句点混淆；
- `balance_check` 做**锚定精确匹配**（`^资产总计$`），避免「流动资产合计」被「资产合计」子串误配；
- 条款抽取只返回「条款类型 + 页码 + 引用」，不照抄长段落。

## 二、`doc_preprocess`：7 阶段知识库管线

```text
discover → classify → extract(后端路由) → clean → structure → output(md+json) → verify
              ↑                                                    ↓
           增量/缓存 ←────────────── 失败人工队列
```

| 阶段 | 做什么 |
|---|---|
| 0 discover | 扫描文件 + 排除（.git / .trash / node_modules 等），记 sha256 |
| 1 classify | 按扩展名 + PDF 文本层检测（>100 字符=文本层，≈0=扫描件）分类 |
| 2 extract | 文本层 PDF→pypdf；扫描件 PDF/图片→OCR；docx→python-docx；xlsx→openpyxl；pptx→python-pptx；csv→markdown 表；json/yaml→代码块 |
| 3 clean | 页眉页脚去重、中文断行合并、数字清洗 |
| 4 structure | 「第X章 / 第X条」→ 标题层级；表格 → markdown 表 |
| 5 output | 每文件 `.md`（YAML frontmatter：source_path/sha256/kind/backend/page_count/ocr_confidence/extracted_at）+ `.json` + `cache.json` |
| 6 verify | 空正文 / 无页 / 不支持格式 → 进 problems 列表 |
| 7 incremental | sha256 命中缓存 → 跳过（`force` 可重跑） |

原始文件永不被修改；markdown 是工作格式，数字另存 JSON；失败进人工队列而不是中断整批。

## 三、运行环境

给一个可用的 Python 安装依赖（RapidOCR 会带上 onnxruntime / opencv / numpy）：

```bash
python3 -m pip install -r requirements.txt
```

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

### 智谱 GLM-OCR 云端引擎

```bash
export ZHIPU_API_KEY=你的key        # 或 DSH_OCR_ZHIPU_KEY / BIGMODEL_API_KEY
export DSH_OCR_ZHIPU_BASE_URL=...   # 可选，默认 https://open.bigmodel.cn/api/paas/v4
```

密钥只从环境变量读取，**不写进代码或配置文件**。云端引擎零第三方依赖（只用 urllib / base64 / json 标准库）。

## 四、安装

从 GitHub：

```bash
dsh plugin add https://github.com/zirscard/dsh-ocr
```

从本地目录：

```bash
dsh plugin add link:/absolute/path/to/dsh-ocr
```

重启 DSH 后，新会话的工具列表出现 `doc_ocr` 与 `doc_preprocess`。

## 五、直接运行引擎（不经过 DSH）

```bash
PY=python3   # 换成能加载 OCR 依赖的解释器

# text：整页文本
"$PY" core/ocr.py --input scan.pdf --mode text --pages 34,37,42 --dpi 200

# table：财务报表（建议 dpi 300–400）
"$PY" core/ocr.py --input scan.pdf --mode table --pages 37 --dpi 300 \
  --out-json /tmp/p37.json --out-md /tmp/p37.md

# clause：合同条款
"$PY" core/ocr.py --input contract-page.png --mode clause

# glm 云端引擎（复杂表格 / 盖章协议优先）
"$PY" core/ocr.py --input contract-page.png --mode table --engine glm

# 知识库预处理
"$PY" core/pipeline.py --input 某目录 --output out/ --mode all
"$PY" core/pipeline.py --input 某目录 --output out/ --mode classify
```

## 六、测试

```bash
python3 test/run_tests.py        # OCR 引擎（纯函数 + 合成图；可选真实样本）
python3 test/glm_mock_test.py    # glm 全链路，用本地 mock 服务器，不花钱
```

- `run_tests.py`：数字清洗、页码解析、合成图关键词命中；设置 `DSH_OCR_TEST_PDF` / `DSH_OCR_TEST_IMAGE` 后追加 table 勾稽与 clause 命中测试。
- `glm_mock_test.py`：本地 mock 智谱 `chat/completions`，验证请求构造（model=glm-ocr / Bearer / data:image;base64）→ Markdown 返回 → 表格解析 → 数字清洗 → BS 勾稽 → 无 Key 时返回结构化错误。

> 仓库不包含任何客户文档或扫描件；真实样本测试只在本地提供路径时运行。

## 七、并发控制

每次 `doc_ocr` 调用都会 spawn 一个独立 Python 进程并重新加载 OCR 模型。并发调用时如果不限流，N 个进程 × onnxruntime 默认「满核线程」= CPU 严重超卖。本插件做了两层限制：

| 层级 | 机制 | 默认 | 覆盖 |
|---|---|---|---|
| 进程内 | `core/ocr.py` 设置 `intra_op_num_threads` / `inter_op_num_threads`（级联 Det/Cls/Rec） | **2** | `DSH_OCR_THREADS`（设 0/负数表示不限） |
| 进程间 | `runner.js` 模块级 FIFO 信号量，限制同时运行的 OCR 进程数 | **2** | `DSH_OCR_MAX_CONCURRENT` |

> 信号量是**同一 DSH host 进程内**的并发上限；跨 host 进程仍会各自 spawn 进程。需要跨进程限制时，设 `DSH_OCR_MAX_CONCURRENT=1` 并在外部串行调用。

## 八、复杂表格的升级路径

本插件的 `table` 模式是「表格结构识别」的**轻量实现**（x 聚类定列 + 列切条）。遇到复杂表格结构、多栏版式、单元格跨行跨列时：

- [PP-Structure 文档分析](https://www.paddleocr.ai/latest/version2.x/ppstructure/overview.html)
- [PP-StructureV3 通用版面解析产线](https://paddlepaddle.github.io/PaddleX/3.0-rc/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html)

RapidOCR 本身就是 PaddleOCR PP-OCRv4 的 ONNX 运行时版（同检测/识别模型），所以升级是「换解析层」而非「换识别引擎」；换到 `glm` 引擎则连解析层一起换成云端方案。

## 九、边界

- 不处理加密 PDF / 需要密码的文档（交给下游）；
- 不做定时任务、不做目录监控；
- 不编造数字：OCR 拿不到或勾稽不平的数，标出来交人工，不硬凑；
- 扫描件/图片才调 OCR；有文本层的 PDF 先用 pypdf 直读；
- `doc_preprocess` 不修改任何原始文件。

## English summary

`dsh-ocr` registers two tools for DeepSeek Harness.

**`doc_ocr`** turns scanned PDFs and images into structured text in three modes: `text` (per-page text and confidence), `table` (financial tables: auto-detected numeric columns, column-slice re-OCR, number cleaning, and a balance-sheet tie-out check), and `clause` (contract-clause keyword hits with page and quote). Two engines are available: `local` runs RapidOCR (PaddleOCR PP-OCRv4 ONNX) entirely on the machine with no network request, and `glm` calls the Zhipu GLM-OCR cloud model and returns native Markdown — stronger on complex tables, formulas and seals. The tool discovers an OCR-capable Python, limits concurrent OCR processes, and reports the exact cells that failed the tie-out instead of inventing numbers.

**`doc_preprocess`** turns a directory of raw files into an Obsidian-ready knowledge base through a seven-stage pipeline: discover, classify, extract (backend routing by file type and PDF text layer), clean, structure, output (markdown with YAML frontmatter plus JSON) and verify, with sha256-based incremental skipping. Original files are never modified.

## License

MIT. RapidOCR / PaddleOCR models and libraries keep their own licenses. The Zhipu GLM-OCR service is used through its public API under Zhipu's own terms.
