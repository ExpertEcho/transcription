"""
Configuration management for SRT Transcript Service.
"""
import os
from typing import Optional
from pathlib import Path


class Settings:
    """Application settings and configuration."""
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # Whisper Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    
    # Audio Processing
    MAX_AUDIO_DURATION: int = int(os.getenv("MAX_AUDIO_DURATION", "7200"))  # 2 hours in seconds
    CHUNK_DURATION: int = int(os.getenv("CHUNK_DURATION", "300"))  # 5 minutes per chunk
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    
    # File Storage
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", "./temp"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "500000000"))  # 500MB
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "3600"))  # 1 hour
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)


# Global settings instance
settings = Settings() 