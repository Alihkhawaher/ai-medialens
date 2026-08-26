"""openrouter_media.py — Analyze video/audio/PDF/image via OpenRouter.

Fills the gap of what Cline can't read: video, audio, and PDF files.
Sends files directly to OpenRouter (/api/v1/chat/completions) which routes
them to multimodal models (and parses PDFs for any model).

Dual-mode:
  CLI : python openrouter_media.py <file> -p "prompt" [--start T] [--end T]
  MCP : launched with NO arguments -> stdio MCP server exposing
        the tool  analyze_media(path, prompt, model?, start?, end?)

Supported inputs (auto-detected by extension):
  Video : .mp4 .mpeg .mov .webm          -> video_url  (base64 data URL)
  Audio : .wav .mp3 .aiff .aac .ogg .flac .m4a -> input_audio (base64)
  PDF   : .pdf                           -> file       (base64 data URL)
  Image : .jpg .jpeg .png .gif .webp     -> image_url  (base64 data URL)

API key resolution order:
  1. OR_KEY environment variable
  2. .env file next to this script (OR_KEY=sk-or-...)

Requires: pip install mcp   (only needed for MCP mode)
"""
import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

API_URL = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_MODEL = 'z-ai/glm-5.3-flash'

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


def get_key():
    return os.environ.get('OR_KEY')


# -------------------------------------------------------------- ffmpeg ----

def find_ffmpeg():
    """Prefer bundled bin/ffmpeg.exe, fall back to PATH."""
    bundled = os.path.join(SCRIPT_DIR, 'bin', 'ffmpeg.exe')
    if os.path.isfile(bundled):
        return bundled
    return shutil.which('ffmpeg')


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


def trim_media(path, start=None, end=None):
    """Trim a video/audio file to [start, end] using ffmpeg.

    Returns path to a trimmed temp file (caller cleans up), or the
    original path if no range was given.
    """
    if not start and not end:
        return path

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found (looked in bin/ and PATH) — "
                           "required for start/end trimming.")

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


def build_payload(path, prompt, model=DEFAULT_MODEL):
    """Build the OpenRouter chat-completions request payload."""
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                build_content_part(path),
                {"type": "text", "text": prompt},
            ]
        }]
    }


def analyze(path, prompt, model=DEFAULT_MODEL, key=None, timeout=300,
            raw=False, start=None, end=None):
    """Send a media file + prompt to OpenRouter and return the analysis.

    Args:
        path:   Path to a video/audio/PDF/image file.
        prompt: Instruction for the model (required).
        model:  OpenRouter model slug.
        key:    API key; defaults to OR_KEY env var / .env file.
        raw:    If True, return full API response dict instead of text.
        start:  Optional trim start ("30", "01:10", "00:01:10").
        end:    Optional trim end (same format).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File not found: {path}')
    key = key or get_key()
    if not key:
        raise RuntimeError('No API key: set OR_KEY in .env or environment')

    trimmed = None
    try:
        trimmed = trim_media(path, start, end)
        payload = build_payload(trimmed, prompt, model)
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


# ------------------------------------------------------------ MCP mode ----

def mcp_run():
    """Run as an MCP stdio server (launched by Cline with no arguments)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("aimedia")

    @mcp.tool()
    def analyze_media(path: str, prompt: str, model: str = DEFAULT_MODEL,
                      start: str = "", end: str = "") -> str:
        """Analyze a video/audio/PDF/image file via OpenRouter.

        Args:
            path: Absolute path to the media file.
            prompt: Instruction for the model (e.g. transcribe, summarize).
            model: OpenRouter model slug.
            start: Optional trim start time ("30", "01:10", "00:01:10").
            end: Optional trim end time (same format). Requires bundled ffmpeg.
        """
        kwargs = {}
        if start:
            kwargs['start'] = start
        if end:
            kwargs['end'] = end
        return analyze(path, prompt, model=model, **kwargs)

    mcp.run()


# ------------------------------------------------------------ CLI mode ----

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Send video/audio/PDF/image to OpenRouter for analysis.")
    ap.add_argument('file', help='Path to the media file')
    ap.add_argument('-p', '--prompt', required=True,
                    help='Instruction for the model')
    ap.add_argument('-m', '--model', default=DEFAULT_MODEL,
                    help=f'OpenRouter model (default: {DEFAULT_MODEL})')
    ap.add_argument('--start', default=None,
                    help='Trim start time, ffmpeg format '
                         '(e.g. 30, 01:10, 00:01:10). Requires ffmpeg.')
    ap.add_argument('--end', default=None,
                    help='Trim end time (same format as --start).')
    ap.add_argument('--dry-run', action='store_true',
                    help='Build the request and show its shape without sending')
    args = ap.parse_args(argv)

    try:
        trimmed = trim_media(args.file, args.start, args.end)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(str(e))

    try:
        if args.dry_run:
            try:
                payload = build_payload(trimmed, args.prompt, args.model)
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
                              start=args.start, end=args.end))
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