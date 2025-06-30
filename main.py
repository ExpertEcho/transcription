"""
FastAPI application entry point for SRT Transcript Service.
"""
import logging
import sys
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger as fastapi_logger
from starlette.responses import Response
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from datetime import datetime
import uvicorn
from config import settings
from models import (
    TranscriptionRequest, TranscriptionResponse, ErrorResponse, HealthResponse
)
from transcriber import transcription_service

# Configure logging
from pythonjsonlogger import jsonlogger
settings.create_directories()
log_handler = logging.FileHandler(settings.LOG_DIR / "server.log")
formatter = jsonlogger.JsonFormatter()
log_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL)
root_logger.addHandler(log_handler)
fastapi_logger.setLevel(settings.LOG_LEVEL)

app = FastAPI(
    title="SRT Transcript Service",
    description="Transcribe long audio files to SRT using Whisper large model (local)",
    version="1.0.0"
)

# CORS (optional, for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    root_logger.info({
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time,
        "timestamp": datetime.utcnow().isoformat()
    })
    return response

@app.post("/transcribe", response_model=TranscriptionResponse, responses={400: {"model": ErrorResponse}})
async def transcribe(
    file: UploadFile = File(..., description="WAV audio file (max 2 hours)"),
    language: str = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Form(False)
):
    """
    Transcribe a WAV audio file and return SRT-formatted transcript.
    """
    start_time = time.time()
    upload_path = None
    try:
        # Save uploaded file
        upload_path = settings.UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Transcribe
        result = await transcription_service.transcribe_file(
            upload_path,
            language=language,
            task=task,
            word_timestamps=word_timestamps
        )
        
        return TranscriptionResponse(
            success=True,
            srt_content=result["srt_content"],
            language=result["language"],
            duration=result["duration"],
            processing_time=result["processing_time"],
            word_count=result["word_count"],
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        root_logger.error(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=str(e),
                error_code="TRANSCRIPTION_ERROR",
                timestamp=datetime.utcnow()
            ).model_dump(mode="json")
        )
    finally:
        # Clean up uploaded file
        if upload_path and upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        model_loaded=transcription_service.is_model_loaded()
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True, workers=settings.WORKERS) 