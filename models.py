"""
Pydantic models for request/response validation.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class TranscriptionRequest(BaseModel):
    """Request model for transcription endpoint."""
    
    language: Optional[str] = Field(
        default=None,
        description="Language code for transcription (e.g., 'en', 'es', 'fr'). If None, auto-detect."
    )
    task: str = Field(
        default="transcribe",
        description="Task type: 'transcribe' or 'translate'",
        pattern="^(transcribe|translate)$"
    )
    word_timestamps: bool = Field(
        default=False,
        description="Include word-level timestamps in SRT output"
    )


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""
    
    success: bool = Field(description="Whether the transcription was successful")
    srt_content: str = Field(description="SRT-formatted transcript")
    language: Optional[str] = Field(description="Detected language code")
    duration: float = Field(description="Audio duration in seconds")
    processing_time: float = Field(description="Total processing time in seconds")
    word_count: int = Field(description="Number of words in transcript")
    timestamp: datetime = Field(description="Processing timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for client handling")
    timestamp: datetime = Field(description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(description="Service status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="Service version")
    model_loaded: bool = Field(description="Whether Whisper model is loaded")


class SRTSubtitle(BaseModel):
    """Individual SRT subtitle entry."""
    
    index: int = Field(description="Subtitle index")
    start_time: str = Field(description="Start time in HH:MM:SS,mmm format")
    end_time: str = Field(description="End time in HH:MM:SS,mmm format")
    text: str = Field(description="Subtitle text content") 