# AI-MediaLens - Media analysis for AI agents via OpenRouter

Fills the gap of what **Cline can't read**: video, audio, and PDF files.
Sends files directly to the OpenRouter API, which routes them to multimodal
models (and parses PDFs for *any* model).

Works as an **MCP server** for Cline, a standalone **CLI**, and an importable
**Python module** - all from one file.

## Quick start

```powershell
git clone https://github.com/Alihkhawaher/ai-medialens.git
cd ai-medialens
pip install mcp          # only needed for MCP mode
python openrouter_media.py setup    # configure API key + register in Cline
```

`setup` interactively:
1. Saves your [OpenRouter key](https://openrouter.ai/keys) to `.env`
2. Registers the `ai-medialens` MCP server into Cline's global config

## Supported inputs (auto-detected by extension)

| Type  | Extensions                                  | OpenRouter content part |
|-------|---------------------------------------------|-------------------------|
| Video | `.mp4` `.mpeg` `.mov` `.webm`               | `video_url` (base64)    |
| Audio | `.wav` `.mp3` `.aiff` `.aac` `.ogg` `.flac` `.m4a` | `input_audio`    |
| PDF   | `.pdf`                                      | `file` (base64)         |
| Image | `.jpg` `.jpeg` `.png` `.gif` `.webp`        | `image_url` (base64)    |

## Usage

### CLI
```powershell
# Full file
python openrouter_media.py video.mp4 -p "Transcribe all spoken Arabic word-for-word"

# Only seconds 5-20 (trimmed locally; ffmpeg auto-downloaded on first use)
python openrouter_media.py video.mp4 -p "What happens here?" --start 5 --end 20

# Different model / explicit key
python openrouter_media.py doc.pdf -p "Summarize" -m z-ai/glm-5.3-flash -k sk-or-v1-...

# Scanned PDF -> real OCR (billed per page by OpenRouter)
python openrouter_media.py scan.pdf -p "Extract all text" --pdf-engine mistral-ocr

# Find models that accept video input, with prices
python openrouter_media.py --list-models video

# Preview request shape without sending (no API cost)
python openrouter_media.py video.mp4 -p "test" --dry-run
```

### MCP (from Cline)
Cline calls the tool natively:
```
use_mcp_tool: server=ai-medialens, tool=analyze_media
arguments: { "path": "...", "prompt": "...", "start": "5", "end": "20" }
```

### As a Python module
```python
from openrouter_media import analyze
text = analyze("video.mp4", "Transcribe all speech.")
resp = analyze("doc.pdf", "Summarize.", raw=True)
```

## API key resolution order

1. `-k/--key` flag
2. `OR_KEY` environment variable
3. `OPENROUTER_API_KEY` environment variable
4. `.env` file next to the script (see .env.example)

## ffmpeg & time-range trimming

`--start`/`--end` trim the media **locally** before upload (OpenRouter has no
native range parameter). ffmpeg is resolved in this order:

1. Bundled `bin\ffmpeg.exe` (if you placed one there)
2. Cached copy at `%LOCALAPPDATA%\ai-medialens\bin\ffmpeg.exe`
3. ffmpeg on PATH
4. **Auto-download** (~80 MB static build, one time, Windows)

Note: trimming uses `-c copy` (fast, but cuts at keyframes; exact cuts would
require re-encoding).

## Models

Default: **`qwen/qwen3.7-flash`** ($0.03/M in, $0.13/M out) — currently the
cheapest OpenRouter model with full video input support.

Other good video-capable options (verify current prices via
`--list-models video`):

| Model | In /M | Out /M |
|---|---|---|
| `qwen/qwen3.7-flash` (default) | $0.03 | $0.13 |
| `z-ai/glm-5.3-flash` | $0.075 | $0.25 |
| `qwen/qwen3.8-27b` | $0.425 | $2.55 |

Any OpenRouter model slug works via `-m` / `model=` — including newer
releases as they appear.

## PDF parsing engines

PDFs are parsed either **locally** or server-side by OpenRouter:

| Engine | Behavior | Cost |
|---|---|---|
| *(default)* | Server-side text extraction from embedded text layer | Free |
| `--pdf-engine local` | **Local parsing via PyMuPDF** (`pip install pymupdf`) — no rate limits. Digital PDFs: text extracted locally. Scanned PDFs: pages rendered as images and the vision model performs the OCR itself | Free |
| `--pdf-engine mistral-ocr` | Server-side real OCR over page images — reads scanned/image-only PDFs | Billed per page |
| `--pdf-engine native` | Forward raw PDF to models with native file input | Free |

`local` is recommended: it never hits OpenRouter's parser rate limits and
handles both digital and scanned PDFs at zero extra cost.

## Notes

- Large files are base64-encoded in-memory - very large videos may hit
  provider upload limits.
- Secrets are safe: `.env` is gitignored and never committed.
