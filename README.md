# AI-MediaLens — Media analysis for AI agents via OpenRouter

Fills the gap of what **Cline can't read**: video, audio, and PDF files.
Sends files directly to the OpenRouter API, which routes them to multimodal
models (and parses PDFs for *any* model).

## Supported inputs (auto-detected by extension)

| Type  | Extensions                                  | OpenRouter content part |
|-------|---------------------------------------------|-------------------------|
| Video | `.mp4` `.mpeg` `.mov` `.webm`               | `video_url` (base64)    |
| Audio | `.wav` `.mp3` `.aiff` `.aac` `.ogg` `.flac` `.m4a` | `input_audio`    |
| PDF   | `.pdf`                                      | `file` (base64)         |
| Image | `.jpg` `.jpeg` `.png` `.gif` `.webp`        | `image_url` (base64)    |

## Setup

1. **API key** — create a `.env` file next to the script:
   ```
   OR_KEY=sk-or-v1-...
   ```
   (The environment variable `OR_KEY` also works and takes precedence.)

2. **MCP mode** (for Cline) — requires:
   ```
   pip install mcp
   ```
   Already registered in Cline's global MCP config as server `ai-medialens`
   with tool `analyze_media`. Restart/reload Cline if it doesn't appear.

3. **ffmpeg** (optional, for time-range trimming) — bundled at `bin\ffmpeg.exe`.
   If missing there, the script falls back to ffmpeg on PATH.

## Usage

### CLI
```powershell
# Full file
python openrouter_media.py video.mp4 -p "Transcribe all spoken Arabic word-for-word"

# Only seconds 5–20 (trimmed locally via bundled ffmpeg)
python openrouter_media.py video.mp4 -p "What happens here?" --start 5 --end 20

# Different model
python openrouter_media.py doc.pdf -p "Summarize" -m google/gemini-2.0-flash-001

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

## Notes

- Time-range trimming uses `-c copy` (fast, but cuts at keyframes; exact cuts
  would require re-encoding).
- Large files are base64-encoded in-memory — very large videos may hit
  provider upload limits.
- The default model is `z-ai/glm-5.3-flash`; override with `-m` / `model=`.