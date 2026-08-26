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
python openrouter_media.py doc.pdf -p "Summarize" -m google/gemini-2.0-flash-001 -k sk-or-v1-...

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

## Notes

- Large files are base64-encoded in-memory - very large videos may hit
  provider upload limits.
- Default model: `google/gemini-2.0-flash-001`; override with `-m`.
- Secrets are safe: `.env` is gitignored and never committed.
