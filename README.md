# WhisperSRT

# SRT Transcript Service

## How to Run This Program

### 1. Install Python dependencies
Make sure you have Python 3.9+ and ffmpeg installed. Then run:
```sh
pip install -r requirements.txt
```

### 1a. (Optional, for GPU) Install PyTorch with CUDA support
If you want to use your NVIDIA GPU for faster transcription, install the CUDA-enabled version of PyTorch **after** installing the requirements:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- Replace `cu121` with your CUDA version if needed (see https://pytorch.org/get-started/locally/).
- **Do not add this line to requirements.txt.**

### 2. (Windows only) Create required folders
In PowerShell or Command Prompt, run:
```sh
mkdir uploads
mkdir temp
mkdir logs
```

### 3. (Optional, for Render.com) Add apt.txt
If deploying to Render, create a file named `apt.txt` with this line:
```
ffmpeg
```

### 4. Run the server
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Open the API docs
Go to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### 6. Upload your audio or video file
- Use the `/transcribe` endpoint in the docs UI to upload your file and get an SRT transcript.
- Or use `curl`:
```sh
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your_audio_or_video.mp4" \
  -F "language=en" \
  -F "task=transcribe" \
  -F "word_timestamps=false"
```

### 7. Check logs
- All requests and inference times are logged in `logs/server.log`.

---

# (rest of the original README follows)

Transcribe long audio files (up to 2+ hours) to SRT subtitles using OpenAI Whisper large model locally. Built with FastAPI, supports async processing, chunked audio, and robust logging.

## Features
- Local Whisper large model (no API key required)
- Accepts HTTP POST requests with WAV (and other) audio files
- Returns SRT-formatted transcript in the response
- Handles long podcast audio files (≥ 2 hours) via chunking
- Uses ffmpeg for audio processing
- Async FastAPI server with request/inference logging

## Requirements
- Python 3.9+
- ffmpeg (must be installed and in PATH)
- CUDA GPU recommended for best performance

## Setup
```bash
# Clone repo and install dependencies
pip install -r requirements.txt

# Ensure ffmpeg is installed
ffmpeg -version
```

## Running the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API Usage
### POST /transcribe
Upload a WAV (or mp3, m4a, etc.) audio file and get SRT transcript.

**Request:**
- `file`: (form-data) Audio file (WAV, mp3, m4a, etc.)
- `language`: (optional, form-data) Language code (e.g. 'en', 'es')
- `task`: (optional, form-data) 'transcribe' or 'translate'
- `word_timestamps`: (optional, form-data) true/false

**Response:**
- `success`: bool
- `srt_content`: SRT-formatted transcript
- `language`: Detected language
- `duration`: Audio duration (seconds)
- `processing_time`: Inference time (seconds)
- `word_count`: Number of words
- `timestamp`: Processing timestamp

**Example (using curl):**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your_audio.wav" \
  -F "language=en" \
  -F "task=transcribe" \
  -F "word_timestamps=false"
```

### GET /health
Health check endpoint.

## Logging
- All requests and inference times are logged to `./logs/server.log` in JSON format.

## Notes
- For best performance, use a CUDA GPU and set `WHISPER_DEVICE=cuda` in your environment.
- The service will automatically chunk long audio files for efficient processing.
- Temporary and uploaded files are cleaned up after processing.

## License
MIT 