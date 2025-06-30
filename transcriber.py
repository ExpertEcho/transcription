"""
Core transcription module using OpenAI Whisper.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import whisper
import torch
from config import settings
from audio_processor import audio_processor

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles Whisper model loading and transcription."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.device = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
            
            # Determine device
            if settings.WHISPER_DEVICE == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = settings.WHISPER_DEVICE
            
            logger.info(f"Using device: {self.device}")
            print(f"[INFO] Whisper will run on: {self.device.upper()}")
            
            # Load model
            self.model = whisper.load_model(
                settings.WHISPER_MODEL,
                device=self.device,
                download_root=None
            )
            
            self.model_loaded = True
            logger.info(f"Successfully loaded Whisper model: {settings.WHISPER_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            self.model_loaded = False
            raise
    
    async def transcribe_audio(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            task: 'transcribe' or 'translate'
            word_timestamps: Include word-level timestamps
            
        Returns:
            Dictionary containing transcription results
        """
        if not self.model_loaded:
            raise RuntimeError("Whisper model not loaded")
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting transcription of {audio_path}")
            
            # Prepare transcription options
            options = {
                "task": task,
                "verbose": False,
                "word_timestamps": word_timestamps
            }
            
            if language:
                options["language"] = language
            
            # Perform transcription
            result = self.model.transcribe(str(audio_path), **options)
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            return {
                "segments": result.get("segments", []),
                "language": result.get("language"),
                "processing_time": processing_time,
                "text": result.get("text", "")
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def segments_to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """
        Convert Whisper segments to SRT format.
        
        Args:
            segments: List of Whisper segments
            
        Returns:
            SRT-formatted string
        """
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            # Format timestamps
            start_timestamp = audio_processor.format_timestamp(start_time)
            end_timestamp = audio_processor.format_timestamp(end_time)
            
            # Add SRT entry
            srt_lines.append(str(i))
            srt_lines.append(f"{start_timestamp} --> {end_timestamp}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries
        
        return "\n".join(srt_lines)
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())


class TranscriptionService:
    """Main transcription service that orchestrates the entire process."""
    
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
    
    async def transcribe_file(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Main transcription method that handles the entire pipeline.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            task: Transcription task
            word_timestamps: Include word timestamps
            
        Returns:
            Transcription results
        """
        async with self.semaphore:
            start_time = time.time()
            temp_files = []
            
            try:
                logger.info(f"Processing transcription request for {audio_path}")
                
                # Validate audio file
                is_valid, error_msg, duration = await audio_processor.validate_audio_file(audio_path)
                if not is_valid:
                    raise ValueError(error_msg)
                
                logger.info(f"Audio file validated: duration={duration}s")
                
                # Convert to WAV if needed
                wav_path = audio_path
                if audio_path.suffix.lower() != '.wav':
                    wav_path = settings.TEMP_DIR / f"{audio_path.stem}_converted.wav"
                    success = await audio_processor.convert_to_wav(audio_path, wav_path)
                    if not success:
                        raise RuntimeError("Failed to convert audio to WAV format")
                    temp_files.append(wav_path)
                
                # Chunk audio if it's long
                chunks = await audio_processor.chunk_audio(wav_path)
                if len(chunks) > 1:
                    temp_files.extend(chunks)
                
                logger.info(f"Processing {len(chunks)} audio chunk(s)")
                
                # Transcribe each chunk
                all_segments = []
                detected_language = None
                
                for i, chunk_path in enumerate(chunks):
                    logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                    
                    chunk_result = await self.transcriber.transcribe_audio(
                        chunk_path,
                        language=language if i == 0 else detected_language,  # Use detected language for subsequent chunks
                        task=task,
                        word_timestamps=word_timestamps
                    )
                    
                    # Store detected language from first chunk
                    if i == 0:
                        detected_language = chunk_result["language"]
                    
                    # Adjust segment timestamps for chunks
                    if len(chunks) > 1:
                        time_offset = i * settings.CHUNK_DURATION
                        for segment in chunk_result["segments"]:
                            segment["start"] += time_offset
                            segment["end"] += time_offset
                    
                    all_segments.extend(chunk_result["segments"])
                
                # Convert to SRT format
                srt_content = self.transcriber.segments_to_srt(all_segments)
                
                # Count words
                full_text = " ".join([seg.get("text", "").strip() for seg in all_segments])
                word_count = self.transcriber.count_words(full_text)
                
                total_time = time.time() - start_time
                
                logger.info(f"Transcription completed successfully: {word_count} words, {total_time:.2f}s")
                
                return {
                    "success": True,
                    "srt_content": srt_content,
                    "language": detected_language,
                    "duration": duration,
                    "processing_time": total_time,
                    "word_count": word_count,
                    "segments": all_segments
                }
                
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                raise
            
            finally:
                # Cleanup temporary files
                if temp_files:
                    await audio_processor.cleanup_temp_files(temp_files)
    
    def is_model_loaded(self) -> bool:
        """Check if Whisper model is loaded."""
        return self.transcriber.model_loaded


# Global transcription service instance
transcription_service = TranscriptionService() 