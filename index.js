/**
 * Host half of dsh-ocr.
 *
 * Registers one agent tool, `doc_ocr`, plus a short system-prompt section that
 * tells the agent when and how to use it. The tool deterministically turns a
 * scanned PDF / image into structured text — plain page text, financial-table
 * rows with a balance-sheet tie-out self-check, or contract-clause hits — so that
 * downstream data pipelines can consume the text layer of documents that have none. All heavy lifting lives in the bundled Python
 * engine (core/ocr.py, RapidOCR = PaddleOCR PP-OCRv4 ONNX); runner.js locates an
 * OCR-capable Python and normalizes its JSON.
 *
 * Keeps ZERO runtime npm dependencies (mirrors dsh-context-extract / dsh-office-audit).
 *
 * @module dsh-ocr
 */
import { fileURLToPath } from 'node:url'
import { dirname, isAbsolute, resolve } from 'node:path'
import { runOcr } from './runner.js'

/** Cordis plugin name. */
export const name = 'dsh-ocr'

/** Required host services: tool registry + prompt assembly. */
export const inject = ['tools', 'systemPrompt']

const HERE = dirname(fileURLToPath(import.meta.url))

/* ------------------------------------------------------------------ */
/* defineTool (structure-compatible reimplementation of the SDK's)     */
/* ------------------------------------------------------------------ */

function compileValue(spec) {
  const node = {}
  if (typeof spec.description === 'string' && spec.description.length > 0) node.description = spec.description
  const type = spec.type
  if (type === undefined || type === 'json') return node
  if (type === 'object') {
    node.type = 'object'
    node.additionalProperties = spec.additionalProperties
    if (spec.properties !== undefined) node.properties = compilePropertyMap(spec.properties).properties
    return node
  }
  if (type === 'array') {
    node.type = 'array'
    if (spec.items !== undefined) node.items = compileValue(spec.items)
    return node
  }
  node.type = type
  if (spec.enum !== undefined) node.enum = [...spec.enum]
  if (spec.const !== undefined) node.const = spec.const
  return node
}

function compilePropertyMap(spec) {
  const properties = {}
  const required = []
  for (const [key, entry] of Object.entries(spec)) {
    const { required: isRequired, ...valueSpec } = entry
    properties[key] = compileValue(valueSpec)
    if (isRequired === true) required.push(key)
  }
  return required.length > 0 ? { properties, required } : { properties }
}

function matchesScalarType(value, type) {
  switch (type) {
    case 'string': return typeof value === 'string'
    case 'number': return typeof value === 'number'
    case 'integer': return typeof value === 'number' && Number.isInteger(value)
    case 'boolean': return typeof value === 'boolean'
    case 'null': return value === null
    default: return true
  }
}

function validateValue(schema, value, path) {
  if (typeof schema.type !== 'string' || schema.type.length === 0) return []
  if (schema.type === 'object') {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return [`${path} must be an object`]
    const violations = []
    for (const key of schema.required ?? []) if (!(key in value)) violations.push(`${path}.${key} is required`)
    if (schema.additionalProperties === false) {
      const known = new Set(Object.keys(schema.properties ?? {}))
      for (const key of Object.keys(value)) if (!known.has(key)) violations.push(`${path}.${key} is not a declared property`)
    }
    for (const [key, child] of Object.entries(schema.properties ?? {})) {
      if (key in value) violations.push(...validateValue(child, value[key], `${path}.${key}`))
    }
    return violations
  }
  if (schema.type === 'array') {
    if (!Array.isArray(value)) return [`${path} must be an array`]
    const violations = []
    if (schema.items !== undefined) value.forEach((item, i) => violations.push(...validateValue(schema.items, item, `${path}[${i}]`)))
    return violations
  }
  if (!matchesScalarType(value, schema.type)) return [`${path} must be ${schema.type}`]
  if (schema.enum !== undefined && !schema.enum.some((v) => v === value)) return [`${path} must be one of ${schema.enum.map(String).join(', ')}`]
  if (schema.const !== undefined && schema.const !== value) return [`${path} must be ${String(schema.const)}`]
  return []
}

function defineTool(options) {
  const compiled = compilePropertyMap(options.parameters)
  const parameters = { type: 'object', properties: compiled.properties }
  if (compiled.required !== undefined) parameters.required = compiled.required
  return {
    name: options.name,
    description: options.description,
    parameters,
    output: {
      schema: {},
      render(args, value) {
        return options.output.render(args, value)
      },
    },
    async execute(args, exec) {
      const violations = validateValue(parameters, args, 'arguments')
      if (violations.length > 0) throw new Error(`Error: invalid arguments: ${violations.join('; ')}`)
      return options.execute(args, exec)
    },
  }
}

/* ------------------------------------------------------------------ */
/* rendering                                                           */
/* ------------------------------------------------------------------ */

function fmt(v) {
  if (v === null || v === undefined) return ''
  if (typeof v === 'number' && Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v))
  return String(v)
}

function renderOcr(_args, value) {
  if (value === null || typeof value !== 'object') return [{ type: 'text', text: String(value) }]
  if (value.error !== undefined || value.ok === false) return [{ type: 'text', text: `OCR 失败：${value.error}` }]

  const lines = []
  const mode = value.mode ?? 'text'
  const head = `引擎 ${value.engine ?? ''} · ${mode} 模式 · ${value.page_count ?? 0} 页 · ${value.elapsed_s ?? 0}s`
  lines.push(`# OCR · ${value.input ?? ''}\n${head}`)

  for (const p of value.pages ?? []) {
    if (mode === 'table' && p.table?.ok) {
      const t = p.table
      lines.push(`\n## 第 ${p.page} 页（表 · ${t.columns} 列）`)
      const header = `| 项目 | ${Array.from({ length: t.columns }, (_, i) => `列${i}`).join(' | ')} |`
      lines.push(header, `|---|${'---|'.repeat(t.columns)}`)
      for (const r of t.rows ?? []) {
        lines.push(`| ${r.label} | ${r.cells.map(fmt).join(' | ')} |`)
      }
      for (const b of t.balance_checks ?? []) {
        lines.push(`> 勾稽 col${b.col}: 资产总计 ${fmt(b.assets_total)} vs 负债+权益 ${fmt(b.liab_plus_equity)} → ${b.balanced ? '✓ 平衡' : '✗ 差 ' + b.diff}`)
      }
    } else if (mode === 'clause') {
      // clauses are emitted once, after the loop
    } else {
      lines.push(`\n## 第 ${p.page} 页（${p.lines} 行 · conf ${p.avg_conf}）`)
      lines.push('```')
      lines.push(p.text)
      lines.push('```')
    }
  }

  if (mode === 'clause' && Array.isArray(value.clauses) && value.clauses.length > 0) {
    lines.push('\n## 条款命中')
    for (const c of value.clauses) {
      lines.push(`- **${c.clause_type}**（${c.page_count_hit} 页，首页 p${c.first_page}）：${(c.quotes?.[0] ?? '').slice(0, 120)}`)
    }
  }

  const meta = []
  if (value.report?.jsonPath) meta.push(`完整 JSON：${value.report.jsonPath}`)
  if (value.report?.mdPath) meta.push(`报告：${value.report.mdPath}`)
  if (meta.length > 0) lines.push('', meta.join(' · '))
  return [{ type: 'text', text: lines.join('\n') }]
}

/* ------------------------------------------------------------------ */
/* plugin apply                                                        */
/* ------------------------------------------------------------------ */

const PROMPT_SECTION = `## OCR 文档识别（doc_ocr）

当材料里出现**扫描件 PDF（无文本层）或图片**（财务尽调报告、审计报告、盖章协议、招股书、表格截图），无法用 pypdf 直接抽文本时，先调 doc_ocr 把它变成可读文本，再交给下游数据处理流程。

- 参数 path：文件路径（.pdf / .png / .jpg …；PDF 会逐页渲染成图再识别）。
- 参数 mode：text（整页文本，最通用）/ table（财务报表：自动定位数值列 + 列切条 + 数字清洗 + BS 平衡勾稽）/ clause（合同条款：回购/对赌/反稀释/优先清算…关键词命中）。
- 参数 pages：可选页范围，如 "all" / "1-10" / "34,37,42"；table 模式建议 dpi 300–400。
- 引擎：RapidOCR（跑 PaddleOCR PP-OCRv4 的 ONNX 模型，中文精度实测 conf 0.95+）。
- 输出：逐页文本 + 平均置信度 + 报告路径；table 模式额外给「资产总计 vs 负债+权益」勾稽结果。

使用纪律：
1. 扫描件/图片才调 OCR；有文本层的 PDF 先用 pypdf 直读，不要无谓 OCR。
2. table 模式只给「能被勾稽验证」的数；勾稽不平的标出来交人工核对，不硬凑。
3. OCR 数字有逗号/句点混淆（52,163 vs 52.163），按千分位规则归一，保留原始文本可复核。
4. 条款抽取只给「条款类型 + 页码 + 引用」，不照抄长段落；回购/对赌窗口要看是否已过期。`

function resolveInputPath(args, exec) {
  let p = args.path
  if (typeof p !== 'string' || p.trim().length === 0) throw new Error('Error: invalid_input: path is required')
  if (!isAbsolute(p)) {
    const cwd = exec?.agent?.session?.header?.cwd
    if (typeof cwd !== 'string' || cwd.length === 0) throw new Error('Error: invalid_input: relative path given but the session has no working directory')
    p = resolve(cwd, p)
  }
  return p
}

export function apply(ctx) {
  const disposers = []

  disposers.push(ctx.tools.register(defineTool({
    name: 'doc_ocr',
    description: 'OCR a scanned PDF or image (no text layer) into structured text. Modes: text (per-page text + confidence), table (financial tables: auto-detect numeric columns, column-slice re-OCR, number cleaning, balance-sheet 勾稽 self-check), clause (contract clause keyword hits with page + quote). Engine: RapidOCR running PaddleOCR PP-OCRv4 ONNX models. Read-only; never modifies the source. For complex table-structure / layout parsing, the upgrade reference is PaddleOCR PP-Structure (paddleocr-doc-parsing).',
    parameters: {
      path: {
        type: 'string',
        required: true,
        description: 'Input file path (.pdf / .png / .jpg / .bmp / .webp / .tif). Absolute, or relative to the session working directory.',
      },
      mode: {
        type: 'string',
        enum: ['text', 'table', 'clause'],
        description: 'OCR mode. text = whole-page text (default). table = financial table extraction with column slicing + balance-sheet 勾稽. clause = contract clause keyword extraction.',
      },
      pages: {
        type: 'string',
        description: 'Page selection: "all" (default), a range "1-10", or a list "34,37,42". Images ignore this.',
      },
      dpi: {
        type: 'integer',
        description: 'Render DPI before OCR. Default 200 (text/clause), 300 (table). Higher DPI (300-400) improves small-number accuracy on scanned financial tables.',
      },
      lang: {
        type: 'string',
        enum: ['ch', 'en'],
        description: 'Recognition language: ch (default) or en.',
      },
      outputDir: {
        type: 'string',
        description: 'Directory for the JSON + Markdown reports (default: <input dir>/.ocr).',
      },
    },
    output: { render: renderOcr },
    async execute(args, exec) {
      const input = resolveInputPath(args, exec)
      const outputDir = typeof args.outputDir === 'string' && args.outputDir.trim().length > 0
        ? (isAbsolute(args.outputDir) ? args.outputDir : resolve(dirname(input), args.outputDir))
        : undefined
      return runOcr({
        input,
        mode: args.mode,
        pages: args.pages,
        dpi: args.dpi,
        lang: args.lang,
        outputDir,
        engineRoot: HERE,
      })
    },
  })))

  const disposeSection = ctx.systemPrompt.section({
    name: 'dsh-ocr',
    order: 187,
    text: PROMPT_SECTION,
  })
  ctx.effect(() => disposeSection, 'dsh-ocr: prompt section')

  ctx.effect(() => () => {
    for (const dispose of disposers.splice(0)) dispose()
  }, 'dsh-ocr: tools')
}
