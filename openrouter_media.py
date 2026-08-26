"""AI-MediaLens — Analyze video/audio/PDF/image via OpenRouter.

Fills the gap of what Cline can't read: video, audio, and PDF files.
Sends files directly to OpenRouter (/api/v1/chat/completions) which routes
them to multimodal models (and parses PDFs for any model).

Modes:
  CLI  : python openrouter_media.py <file> -p "prompt" [--start T] [--end T]
  Setup: python openrouter_media.py setup        # configure key + Cline MCP
  Models: python openrouter_media.py --list-models [video|audio]
  MCP  : launched with NO arguments -> stdio MCP server exposing
         the tool  analyze_media(path, prompt, model?, start?, end?)

Supported inputs (auto-detected by extension):
  Video : .mp4 .mpeg .mov .webm          -> video_url  (base64 data URL)
  Audio : .wav .mp3 .aiff .aac .ogg .flac .m4a -> input_audio (base64)
  PDF   : .pdf                           -> file       (base64 data URL)
  Image : .jpg .jpeg .png .gif .webp     -> image_url  (base64 data URL)

API key resolution order:
  1. --key flag          2. OR_KEY env var        3. OPENROUTER_API_KEY env var
  4. .env file next to this script

Requires: pip install mcp   (only needed for MCP mode)
"""
import argparse
import base64
import io
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

API_URL = 'https://openrouter.ai/api/v1/chat/completions'
MODELS_URL = 'https://openrouter.ai/api/v1/models'
DEFAULT_MODEL = 'qwen/qwen3.7-flash'

# Server-side PDF parsing engines offered by OpenRouter's file-parser plugin.
# - cloudflare-ai : default text extraction (free; fails on scanned PDFs)
# - mistral-ocr   : real OCR over page images — reads scanned PDFs
#                   (billed per page by OpenRouter)
# - native        : forward raw PDF to models with native file input
# Plus a fully LOCAL engine (no rate limits, no parsing cost):
# - local         : PyMuPDF extracts the text layer locally; for scanned
#                   PDFs (no text layer) pages are rendered to images and
#                   the vision model performs the OCR itself.
SERVER_PDF_ENGINES = ('cloudflare-ai', 'mistral-ocr', 'native')
PDF_ENGINES = SERVER_PDF_ENGINES + ('local',)

FFMPEG_URL = ('https://www.gyan.dev/ffmpeg/builds/'
              'ffmpeg-release-essentials.zip')
CACHE_DIR = os.path.join(os.environ.get('LOCALAPPDATA',
                                        os.path.expanduser('~')),
                         'ai-medialens', 'bin')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_EXTS = {'.mp4', '.mpeg', '.mov', '.webm'}
AUDIO_EXTS = {'.wav', '.mp3', '.aiff', '.aac', '.ogg', '.flac', '.m4a'}
PDF_EXTS = {'.pdf'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

SUPPORTED_EXTS = VIDEO_EXTS | AUDIO_EXTS | PDF_EXTS | IMAGE_EXTS


# ---------------------------------------------------------------- .env ----

def load_env():
    """Load KEY=VALUE pairs from a .env file next to this script into
    os.environ (without overriding existing variables)."""
    env_path = os.path.join(SCRIPT_DIR, '.env')
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


load_env()


def get_key(explicit=None):
    """Resolve API key: explicit flag > OR_KEY > OPENROUTER_API_KEY."""
    return (explicit or os.environ.get('OR_KEY')
            or os.environ.get('OPENROUTER_API_KEY'))


KEY_ERROR = ("No API key found.\n"
             "  Get one free at https://openrouter.ai/keys then either:\n"
             "  - run:  python openrouter_media.py setup\n"
             "  - or set OR_KEY / OPENROUTER_API_KEY environment variable")


# -------------------------------------------------------------- ffmpeg ----

def find_ffmpeg():
    """Locate ffmpeg: bundled bin/ -> PATH -> auto-download cache dir."""
    bundled = os.path.join(SCRIPT_DIR, 'bin', 'ffmpeg.exe')
    if os.path.isfile(bundled):
        return bundled
    cached = os.path.join(CACHE_DIR, 'ffmpeg.exe')
    if os.path.isfile(cached):
        return cached
    return shutil.which('ffmpeg')


def download_ffmpeg():
    """Download a static ffmpeg build once into the user cache dir."""
    target = os.path.join(CACHE_DIR, 'ffmpeg.exe')
    if os.path.isfile(target):
        return target
    if sys.platform != 'win32':
        raise RuntimeError("Automatic ffmpeg download supports Windows only. "
                           "Install ffmpeg via your package manager.")
    print(f"[setup] Downloading ffmpeg (~80 MB, one time) ...", file=sys.stderr)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_zip = os.path.join(CACHE_DIR, 'ffmpeg.zip')
    try:
        urllib.request.urlretrieve(FFMPEG_URL, tmp_zip)
        with zipfile.ZipFile(tmp_zip) as z:
            member = next(n for n in z.namelist()
                          if n.endswith('/bin/ffmpeg.exe'))
            with z.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    finally:
        if os.path.isfile(tmp_zip):
            os.unlink(tmp_zip)
    print(f"[setup] ffmpeg saved to {target}", file=sys.stderr)
    return target


def trim_media(path, start=None, end=None):
    """Trim a video/audio file to [start, end] using ffmpeg.

    Auto-downloads ffmpeg on first use if none is installed.
    Returns path to a trimmed temp file (caller cleans up), or the
    original path if no range was given.
    """
    if not start and not end:
        return path

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        ffmpeg = download_ffmpeg()

    ext = os.path.splitext(path)[1].lower()
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.close()

    cmd = [ffmpeg, '-y', '-loglevel', 'error']
    if start:
        cmd += ['-ss', start]           # seek before input = fast seek
    if end:
        cmd += ['-to', end]
    cmd += ['-i', path, '-c', 'copy', tmp.name]

    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.unlink(tmp.name)
        err = e.stderr.decode(errors='replace')[:500]
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}): {err}. "
                           "Note: -c copy trims at keyframes; re-encode may "
                           "be needed for exact cuts.") from e
    return tmp.name


# ----------------------------------------------------------- core logic ---

def b64_of(path):
    """Return base64-encoded contents of a file."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def build_content_part(path):
    """Detect file type by extension and return the matching content part."""
    ext = os.path.splitext(path)[1].lower()
    mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'

    if ext in VIDEO_EXTS:
        b64 = b64_of(path)
        return {"type": "video_url",
                "video_url": {"url": f"data:{mime};base64,{b64}"}}
    if ext in AUDIO_EXTS:
        # audio must be base64 WITHOUT the data-url prefix
        b64 = b64_of(path)
        fmt = {'.mp3': 'mp3', '.wav': 'wav', '.aiff': 'aiff', '.aac': 'aac',
               '.ogg': 'ogg', '.flac': 'flac', '.m4a': 'm4a'}[ext]
        return {"type": "input_audio",
                "input_audio": {"data": b64, "format": fmt}}
    if ext in PDF_EXTS:
        b64 = b64_of(path)
        return {"type": "file",
                "file": {"filename": os.path.basename(path),
                         "file_data": f"data:application/pdf;base64,{b64}"}}
    if ext in IMAGE_EXTS:
        b64 = b64_of(path)
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}}

    raise ValueError(f"Unsupported file type '{ext}'. Supported: "
                     f"{sorted(SUPPORTED_EXTS)}")


def pdf_local_content(path, max_pages=10):
    """Parse a PDF locally with PyMuPDF into message content parts.

    - Digital PDFs: extracted text is returned as a single text part.
    - Scanned PDFs (no/embedded-empty text layer): pages are rendered to
      PNG images so the vision model can read them directly (free OCR).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "Local PDF parsing requires PyMuPDF:\n  pip install pymupdf")

    doc = fitz.open(path)
    n_pages = min(doc.page_count, max_pages)
    text = "\n".join(doc[i].get_text() for i in range(n_pages))

    if len(text.strip()) >= 50 * max(n_pages, 1):
        # Real text layer present — send it directly.
        return [{"type": "text",
                 "text": f"[Extracted PDF text, {n_pages} page(s)]\n{text}"}]

    # Little/no text -> scanned document; render pages as images.
    parts = []
    for i in range(n_pages):
        pix = doc[i].get_pixmap(dpi=110)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{b64}"}})
    note = (f"[Scanned PDF, {n_pages} of {doc.page_count} page(s) rendered "
            f"as images. Read them with OCR.]")
    return parts + [{"type": "text", "text": note}]


def build_payload(path, prompt, model=DEFAULT_MODEL, pdf_engine=None):
    """Build the OpenRouter chat-completions request payload.

    Args:
        pdf_engine: Optional SERVER-side PDF parsing engine for
                    OpenRouter's file-parser plugin. One of
                    SERVER_PDF_ENGINES, or None for OpenRouter's default.
                    ('local' is handled separately in analyze().)
    """
    if pdf_engine is not None and pdf_engine not in SERVER_PDF_ENGINES:
        raise ValueError(f"Invalid pdf_engine '{pdf_engine}'. "
                         f"Choose from: {PDF_ENGINES}")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                build_content_part(path),
                {"type": "text", "text": prompt},
            ]
        }]
    }
    if pdf_engine:
        payload["plugins"] = [{"id": "file-parser",
                               "pdf": {"engine": pdf_engine}}]
    return payload


def analyze(path, prompt, model=DEFAULT_MODEL, key=None, timeout=300,
            raw=False, start=None, end=None, pdf_engine=None):
    """Send a media file + prompt to OpenRouter and return the analysis.

    Args:
        path:   Path to a video/audio/PDF/image file.
        prompt: Instruction for the model (required).
        model:  OpenRouter model slug.
        key:    API key; defaults to OR_KEY / OPENROUTER_API_KEY / .env.
        raw:    If True, return full API response dict instead of text.
        start:  Optional trim start ("30", "01:10", "00:01:10").
        end:    Optional trim end (same format).
        pdf_engine: Optional PDF parsing engine (see PDF_ENGINES). Use
                    'mistral-ocr' for scanned/image-only PDFs.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File not found: {path}')
    key = get_key(key)
    if not key:
        raise RuntimeError(KEY_ERROR)

    trimmed = None
    try:
        trimmed = trim_media(path, start, end)

        is_pdf = os.path.splitext(trimmed)[1].lower() in PDF_EXTS
        if pdf_engine == 'local' and is_pdf:
            # Fully local parsing — bypasses OpenRouter's rate-limited
            # file-parser entirely.
            parts = pdf_local_content(trimmed)
            payload = {
                "model": model,
                "messages": [{"role": "user",
                              "content": parts +
                                         [{"type": "text", "text": prompt}]}]
            }
        else:
            payload = build_payload(trimmed, prompt, model,
                                    pdf_engine=pdf_engine)
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode(),
            headers={'Authorization': f'Bearer {key}',
                     'Content-Type': 'application/json'})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:2000]}')

        return resp if raw else resp['choices'][0]['message']['content']
    finally:
        if trimmed and trimmed != path:
            os.unlink(trimmed)


# -------------------------------------------------------- list models ----

def list_models(modality=None):
    """Print OpenRouter models supporting the given input modality."""
    with urllib.request.urlopen(MODELS_URL, timeout=60) as r:
        data = json.loads(r.read())
    rows = []
    for m in data.get('data', []):
        arch = m.get('architecture', {})
        inputs = arch.get('input_modalities') or []
        if modality and modality not in inputs:
            continue
        price = m.get('pricing', {}).get('prompt', '?')
        try:
            price = f"${float(price) * 1_000_000:.2f}/M"
        except (TypeError, ValueError):
            pass
        rows.append((m['id'], ','.join(inputs), price))
    rows.sort()
    print(f"{'MODEL':<50} {'INPUTS':<25} PRICE")
    for mid, inputs, price in rows:
        print(f"{mid:<50} {inputs:<25} {price}")
    print(f"\n{len(rows)} model(s)")


# ------------------------------------------------------------ setup ------

CLINE_SETTINGS = os.path.join(
    os.environ.get('APPDATA', os.path.expanduser('~')),
    'Code', 'User', 'globalStorage', 'saoudrizwan.claude-dev',
    'settings', 'cline_mcp_settings.json')

SERVER_NAME = 'ai-medialens'


def cmd_setup():
    """Interactive one-shot configuration: API key + Cline MCP registration."""
    # --- 1. API key ---
    env_path = os.path.join(SCRIPT_DIR, '.env')
    if get_key():
        print("[1/2] API key: already configured (OR_KEY / OPENROUTER_API_KEY "
              "/ .env). Skipping.")
    else:
        key = input("Paste your OpenRouter key "
                    "(get one at https://openrouter.ai/keys): ").strip()
        if not key:
            sys.exit("No key entered. Aborting.")
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(f"OR_KEY={key}\n")
        print(f"[1/2] API key saved to {env_path}")

    # --- 2. Cline MCP registration ---
    if not os.path.isfile(CLINE_SETTINGS):
        print(f"[2/2] Cline settings not found at:\n  {CLINE_SETTINGS}\n"
              "Add this manually to your mcpServers config:")
        print(json.dumps({SERVER_NAME: _mcp_entry()}, indent=2))
        return
    with open(CLINE_SETTINGS, encoding='utf-8') as f:
        settings = json.load(f)
    servers = settings.setdefault('mcpServers', {})
    if SERVER_NAME in servers:
        print(f"[2/2] Cline MCP server '{SERVER_NAME}' already registered. "
              "Skipping.")
        return
    ans = input(f"[2/2] Register '{SERVER_NAME}' in Cline's global MCP "
                "config? [y/N]: ").strip().lower()
    if ans == 'y':
        servers[SERVER_NAME] = _mcp_entry()
        with open(CLINE_SETTINGS, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        print(f"      Registered. Restart/reload Cline to activate.")
    else:
        print("      Skipped.")


def _mcp_entry():
    return {
        "autoApprove": [],
        "disabled": False,
        "timeout": 300,
        "type": "stdio",
        "command": sys.executable,
        "args": [os.path.join(SCRIPT_DIR, 'openrouter_media.py')],
        "env": {},
    }


# ------------------------------------------------------------ MCP mode ----

def mcp_run():
    """Run as an MCP stdio server (launched by Cline with no arguments)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("ai-medialens")

    @mcp.tool()
    def analyze_media(path: str, prompt: str, model: str = DEFAULT_MODEL,
                      start: str = "", end: str = "",
                      pdf_engine: str = "") -> str:
        """Analyze a video/audio/PDF/image file via OpenRouter.

        Args:
            path: Absolute path to the media file.
            prompt: Instruction for the model (e.g. transcribe, summarize).
            model: OpenRouter model slug.
            start: Optional trim start time ("30", "01:10", "00:01:10").
            end: Optional trim end time (same format). Requires ffmpeg
                 (auto-downloaded on first use on Windows).
            pdf_engine: For PDFs only. 'local' parses locally with PyMuPDF
                (no rate limits; renders scanned pages as images for the
                vision model to OCR). Server-side options: cloudflare-ai |
                mistral-ocr (real OCR for scans, billed per page) | native.
                Empty = OpenRouter default parser.
        """
        kwargs = {}
        if start:
            kwargs['start'] = start
        if end:
            kwargs['end'] = end
        if pdf_engine:
            kwargs['pdf_engine'] = pdf_engine
        return analyze(path, prompt, model=model, **kwargs)

    mcp.run()


# ------------------------------------------------------------ CLI mode ----

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="openrouter_media.py",
        description="AI-MediaLens: send video/audio/PDF/image to OpenRouter "
                    "for analysis (fills gaps Cline doesn't support).")
    ap.add_argument('file', nargs='?', help='Path to the media file')
    ap.add_argument('-p', '--prompt', help='Instruction for the model')
    ap.add_argument('-k', '--key', default=None,
                    help='OpenRouter API key (else OR_KEY / '
                         'OPENROUTER_API_KEY / .env)')
    ap.add_argument('-m', '--model', default=DEFAULT_MODEL,
                    help=f'OpenRouter model (default: {DEFAULT_MODEL})')
    ap.add_argument('--start', default=None,
                    help='Trim start time, ffmpeg format '
                         '(e.g. 30, 01:10, 00:01:10).')
    ap.add_argument('--end', default=None,
                    help='Trim end time (same format as --start).')
    ap.add_argument('--pdf-engine', default=None, choices=PDF_ENGINES,
                    help="PDF parsing engine (OpenRouter file-parser "
                         "plugin). 'mistral-ocr' reads scanned/image-only "
                         "PDFs via real OCR (billed per page).")
    ap.add_argument('--list-models', nargs='?', const='all',
                    metavar='MODALITY',
                    help="List OpenRouter models, optionally filtered by "
                         "input modality: video | audio | image | text")
    ap.add_argument('--dry-run', action='store_true',
                    help='Build the request and show its shape without sending')
    args = ap.parse_args(argv)

    # Sub-command style helpers -----------------------------------------
    if args.file == 'setup':
        cmd_setup()
        return

    if args.list_models:
        list_models(None if args.list_models == 'all' else args.list_models)
        return

    # Normal analysis flow -----------------------------------------------
    if not args.file:
        ap.error('the following arguments are required: file '
                 '(or use "setup" / "--list-models")')
    if not args.prompt:
        ap.error('-p/--prompt is required')

    try:
        trimmed = trim_media(args.file, args.start, args.end)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(str(e))

    try:
        if args.dry_run:
            try:
                payload = build_payload(trimmed, args.prompt, args.model,
                                        pdf_engine=args.pdf_engine)
            except (FileNotFoundError, ValueError) as e:
                sys.exit(str(e))
            part = payload['messages'][0]['content'][0]
            shape = {'type': part['type']}
            for k, v in part.items():
                if k != 'type':
                    shape[k] = (list(v.keys()) if isinstance(v, dict)
                                else f'<{len(str(v))} chars>')
            print(json.dumps({'model': payload['model'],
                              'content_part': shape}, indent=2))
        else:
            try:
                print(analyze(args.file, args.prompt, args.model,
                              key=args.key,
                              start=args.start, end=args.end,
                              pdf_engine=args.pdf_engine))
            except (RuntimeError, FileNotFoundError) as e:
                sys.exit(str(e))
    finally:
        if trimmed != args.file:
            os.unlink(trimmed)


if __name__ == '__main__':
    # No arguments -> MCP stdio server mode; with args -> CLI mode.
    if len(sys.argv) > 1:
        main()
    else:
        mcp_run()