/**
 * Runtime bridge for dsh-ocr.
 *
 * OCR needs RapidOCR (onnxruntime / opencv / pypdfium2) — all C-extension wheels
 * that the DSH bundled Python cannot load under macOS hardened-runtime library
 * validation (dlopen: "mapping process and mapped file have different Team IDs").
 * So instead of `findPython()` (DSH python), this runner locates an OCR-capable
 * interpreter — an explicit `DSH_OCR_PYTHON` override, then Anaconda, then the DSH
 * python as a last resort — and, for non-DSH interpreters, appends the DSH
 * site-packages (where rapidocr_onnxruntime is installed) via the bundled
 * `core/_boot/sitecustomize.py`, mirroring the proven `toolchain/v3py` mechanism.
 *
 * No external npm dependencies. A missing OCR interpreter is a structured error
 * with a remediation hint, not a crash.
 *
 * @module dsh-ocr/runner
 */
import { spawn } from 'node:child_process'
import { existsSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, extname, isAbsolute, join, resolve } from 'node:path'

function dshHome() {
  const override = process.env.DSH_HOME
  return (override !== undefined && override.length > 0 ? override : join(homedir(), '.dsh'))
}

/** DSH primary-runtime site-packages (holds rapidocr_onnxruntime and its C wheels). */
export function dshSitePackages() {
  const runtime = join(dshHome(), 'dsh-runtimes', 'dsh-primary-runtime', 'dependencies', 'python')
  const candidates = [
    join(runtime, 'lib', 'python3.12', 'site-packages'),
    join(runtime, 'lib', 'python3.11', 'site-packages'),
    join(runtime, 'lib', 'python3.13', 'site-packages'),
    join(runtime, 'Lib', 'site-packages'),
  ]
  for (const c of candidates) {
    try {
      if (existsSync(c) && statSync(c).isDirectory()) return c
    } catch { /* ignore */ }
  }
  return undefined
}

function firstExistingFile(candidates) {
  for (const c of candidates) {
    if (c === undefined || c.length === 0) continue
    try {
      if (existsSync(c) && statSync(c).isFile()) return c
    } catch { /* ignore */ }
  }
  return undefined
}

function dshPython() {
  const runtime = join(dshHome(), 'dsh-runtimes', 'dsh-primary-runtime', 'dependencies', 'python')
  return firstExistingFile([
    join(runtime, 'bin', 'python3'),
    join(runtime, 'bin', 'python'),
    join(runtime, 'python.exe'),
    join(runtime, 'bin', 'python.exe'),
  ]) ?? 'python3'
}

/**
 * Ordered list of OCR-capable Python candidates.
 *
 * 1. `DSH_OCR_PYTHON` explicit override.
 * 2. DSH built-in Python — preferred since the macOS sign-fix (disable-library-
 *    validation) makes it load onnxruntime/cv2/pypdfium2. See dsh-macos-signfix.
 * 3. Anaconda `/opt/anaconda3/bin/python3` — fallback for machines where the
 *    built-in interpreter was re-signed by a DSH update and the fix is pending.
 *
 * Each candidate carries `isDsh` (whether to append DSH site-packages via the
 * bundled _boot/sitecustomize.py) and `source` (for diagnostics).
 */
export function findOcrPythonCandidates() {
  const dsh = dshPython()
  const list = []
  const explicit = process.env.DSH_OCR_PYTHON
  if (explicit && explicit.length > 0) list.push({ python: explicit, isDsh: false, source: 'env' })
  if (existsSync(dsh)) list.push({ python: dsh, isDsh: true, source: 'dsh-builtin' })
  const anaconda = '/opt/anaconda3/bin/python3'
  if (existsSync(anaconda) && statSync(anaconda).isFile()) list.push({ python: anaconda, isDsh: false, source: 'anaconda' })
  // dedupe by resolved path, keep first occurrence
  const seen = new Set()
  return list.filter((c) => {
    const key = c.python
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

/** Back-compat single-pick helper (first candidate). */
export function findOcrPython() {
  const list = findOcrPythonCandidates()
  return list[0] ?? { python: dshPython(), isDsh: true, source: 'dsh-builtin' }
}

// ---- process-level concurrency limiter ---------------------------------
// Each doc_ocr call spawns a fresh Python + RapidOCR process. A burst of N
// parallel tool calls therefore starts N model loads + N inference processes.
// Cap how many run at once so the CPU is not oversubscribed and one call does
// not appear to hang; excess calls queue FIFO. Override with
// DSH_OCR_MAX_CONCURRENT (default 2).
const MAX_CONCURRENT = Math.max(1, Number(process.env.DSH_OCR_MAX_CONCURRENT || 2))
let activeProcs = 0
const waitQueue = []

function acquireSlot() {
  if (activeProcs < MAX_CONCURRENT) {
    activeProcs += 1
    return Promise.resolve()
  }
  return new Promise((resolve) => waitQueue.push(resolve))
}

function releaseSlot() {
  const next = waitQueue.shift()
  if (next) {
    next()
    return
  }
  activeProcs = Math.max(0, activeProcs - 1)
}

function exec(command, args, options = {}) {
  return new Promise((resolvePromise) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      // PYTHONUTF8/PYTHONIOENCODING force the engine's stdout to UTF-8: on Windows
      // a piped stdout defaults to the locale code page (cp936 here), which turned
      // the Chinese text in the JSON payload into mojibake while the --out-json /
      // --out-md files written by the engine itself stayed correct.
      env: { ...process.env, ...(options.env ?? {}), PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    // Decode as UTF-8 explicitly and let the stream own chunk boundaries, so a
    // multi-byte character split across two chunks is not turned into U+FFFD.
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    let stdout = ''
    let stderr = ''
    let settled = false
    const timeoutMs = options.timeoutMs ?? 600_000
    const timer = setTimeout(() => {
      if (settled) return
      settled = true
      child.kill('SIGKILL')
      resolvePromise({ code: -1, stdout, stderr: `${stderr}\n[timeout after ${timeoutMs}ms]`, timedOut: true })
    }, timeoutMs)
    child.stdout.on('data', (chunk) => { stdout += String(chunk) })
    child.stderr.on('data', (chunk) => { stderr += String(chunk) })
    child.on('error', (error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolvePromise({ code: -1, stdout, stderr: `${stderr}\n${String(error?.message ?? error)}`, spawnError: error })
    })
    child.on('close', (code) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolvePromise({ code: code ?? -1, stdout, stderr })
    })
  })
}

function timestamp() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
}

function normalizePages(pages) {
  if (pages === undefined || pages === null) return 'all'
  if (Array.isArray(pages)) return pages.map(String).join(',')
  if (typeof pages === 'number') return String(pages)
  return String(pages)
}

/**
 * Run OCR over a scanned PDF or image.
 *
 * @param {object} options
 * @param {string} options.input - absolute input path (.pdf / .png / .jpg ...)
 * @param {string} [options.mode] - "text" | "table" | "clause" (default "text")
 * @param {string|number|Array} [options.pages] - page range "all" / "1-10" / "34,37,42"
 * @param {number} [options.dpi] - render DPI (default 200; table mode ≈ 300-400)
 * @param {string} [options.lang] - "ch" | "en"
 * @param {string} [options.outputDir] - report directory
 * @param {string} options.engineRoot - plugin root (contains core/ocr.py)
 */
export async function runOcr(options) {
  const input = resolve(options.input)
  if (!existsSync(input)) return { ok: false, error: `input not found: ${input}` }

  const engine = join(options.engineRoot, 'core', 'ocr.py')
  if (!existsSync(engine)) return { ok: false, error: `OCR engine missing: ${engine}` }

  const bootDir = join(options.engineRoot, 'core', '_boot')
  const extraSite = dshSitePackages()

  const ext = extname(input).toLowerCase()
  const mode = (options.mode ?? 'text') || 'text'
  const reportDir = options.outputDir ?? join(dirname(input), '.ocr')
  const stem = basename(input, ext)
  const stamp = timestamp()
  const jsonPath = join(reportDir, `${stem}-${mode}-${stamp}.json`)
  const mdPath = join(reportDir, `${stem}-${mode}-${stamp}.md`)

  const args = [
    engine,
    '--input', input,
    '--mode', mode,
    '--pages', normalizePages(options.pages),
    '--dpi', String(options.dpi ?? (mode === 'table' ? 300 : 200)),
    '--lang', options.lang ?? 'ch',
    '--engine', options.engine ?? 'local',
    '--out-json', jsonPath,
    '--out-md', mdPath,
  ]
  if (options.glmPrompt) args.push('--glm-prompt', String(options.glmPrompt))

  const candidates = findOcrPythonCandidates()
  const tried = []
  let lastError = null

  for (const cand of candidates) {
    // DSH built-in python already has its own site-packages first; non-DSH
    // interpreters need the DSH site-packages appended via _boot/sitecustomize.py.
    const env = {}
    if (!cand.isDsh) {
      if (extraSite) env.V3_EXTRA_SITE = extraSite
      env.PYTHONPATH = [bootDir, process.env.PYTHONPATH].filter(Boolean).join(process.platform === 'win32' ? ';' : ':')
    }
    const result = await (async () => {
      await acquireSlot()
      try {
        return await exec(cand.python, args, { cwd: dirname(engine), env, timeoutMs: 600_000 })
      } finally {
        releaseSlot()
      }
    })()

    if (result.code !== 0) {
      const detail = (result.stderr || result.stdout || '').trim().split('\n').slice(-8).join('\n')
      tried.push({ source: cand.source, python: cand.python, error: `exit ${result.code}: ${detail.slice(0, 300)}` })
      lastError = detail
      continue
    }
    let payload
    try {
      payload = JSON.parse(result.stdout)
    } catch {
      tried.push({ source: cand.source, python: cand.python, error: 'invalid JSON from engine' })
      lastError = 'invalid JSON from engine'
      continue
    }
    // engine_unavailable → this interpreter cannot load the native extensions;
    // try the next candidate (e.g. fall back to Anaconda).
    if (payload && payload.ok === false && payload.error === 'engine_unavailable') {
      tried.push({ source: cand.source, python: cand.python, error: payload.detail })
      lastError = payload.detail
      continue
    }
    if (payload && typeof payload === 'object' && payload.ok) {
      payload.report = { jsonPath, mdPath }
      payload.interpreter = cand.source
      if (tried.length > 0) payload.fallback_used = tried
      return payload
    }
    // other structured error (e.g. PDF open failed) → return as-is
    payload.interpreter = cand.source
    return payload
  }

  return {
    ok: false,
    error: `no OCR-capable interpreter succeeded (tried: ${tried.map((t) => t.source).join(', ') || 'none'})${lastError ? `: ${lastError}` : ''}`,
    tried,
    hint: 'Run the macOS sign fix (dsh-macos-signfix, doc_ocr_sign_fix mode=auto) to make the DSH built-in Python load onnxruntime/cv2/pypdfium2, or set DSH_OCR_PYTHON to an Anaconda/conda python3 with rapidocr-onnxruntime installed.',
  }
}

/**
 * Run the 7-stage knowledge-base preprocessing pipeline over a file or directory.
 *
 * Stages: discover → classify → extract → clean → structure → output → verify,
 * with sha256-based incremental skip. Outputs Obsidian-flavored markdown
 * (YAML frontmatter) + JSON + a cache.json under outputDir.
 *
 * @param {object} options
 * @param {string} options.input - file or directory to preprocess
 * @param {string} [options.outputDir] - output directory (default <input>/.preprocessed)
 * @param {'all'|'classify'|'discover'} [options.mode]
 * @param {number} [options.dpi] - OCR render DPI (default 200)
 * @param {number} [options.maxFiles] - cap files processed (default 200)
 * @param {boolean} [options.force] - ignore cache, reprocess all
 * @param {string} options.engineRoot - plugin root (contains core/pipeline.py)
 */
export async function runPreprocess(options) {
  const input = resolve(options.input)
  if (!existsSync(input)) return { ok: false, error: `input not found: ${input}` }

  const engine = join(options.engineRoot, 'core', 'pipeline.py')
  if (!existsSync(engine)) return { ok: false, error: `pipeline engine missing: ${engine}` }

  const mode = options.mode ?? 'all'
  const outputDir = options.outputDir ?? join(dirname(input), '.preprocessed')

  const args = [
    engine,
    '--input', input,
    '--output', outputDir,
    '--mode', mode,
    '--dpi', String(options.dpi ?? 200),
    '--max-files', String(options.maxFiles ?? 200),
    '--engine', options.ocrEngine ?? 'local',
  ]
  if (options.force) args.push('--force')

  const candidates = findOcrPythonCandidates()
  const bootDir = join(options.engineRoot, 'core', '_boot')
  const extraSite = dshSitePackages()
  const tried = []
  let lastError = null

  for (const cand of candidates) {
    const env = {}
    if (!cand.isDsh) {
      if (extraSite) env.V3_EXTRA_SITE = extraSite
      env.PYTHONPATH = [bootDir, process.env.PYTHONPATH].filter(Boolean).join(process.platform === 'win32' ? ';' : ':')
    }
    const result = await (async () => {
      await acquireSlot()
      try {
        return await exec(cand.python, args, { cwd: dirname(engine), env, timeoutMs: 1_800_000 })
      } finally {
        releaseSlot()
      }
    })()
    if (result.code !== 0) {
      const detail = (result.stderr || result.stdout || '').trim().split('\n').slice(-8).join('\n')
      tried.push({ source: cand.source, error: `exit ${result.code}: ${detail.slice(0, 300)}` })
      lastError = detail
      continue
    }
    let payload
    try {
      payload = JSON.parse(result.stdout)
    } catch {
      tried.push({ source: cand.source, error: 'invalid JSON from pipeline' })
      lastError = 'invalid JSON from pipeline'
      continue
    }
    if (payload && typeof payload === 'object' && payload.ok) {
      payload.interpreter = cand.source
      if (tried.length > 0) payload.fallback_used = tried
      return payload
    }
    tried.push({ source: cand.source, error: payload?.error ?? 'pipeline returned ok=false' })
    lastError = payload?.error ?? 'pipeline returned ok=false'
  }

  return {
    ok: false,
    error: `pipeline failed on all interpreters (tried: ${tried.map((t) => t.source).join(', ') || 'none'})${lastError ? `: ${lastError}` : ''}`,
    tried,
    hint: 'Check that the OCR interpreter can import pypdf/pdfplumber/openpyxl/python-docx/python-pptx/rapidocr. Run doc_ocr_sign_fix(mode="auto") if native extensions fail to load.',
  }
}
