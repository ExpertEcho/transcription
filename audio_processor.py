"""
Audio processing utilities for SRT Transcript Service.
"""
import os
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import ffmpeg
import librosa
import soundfile as sf
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing, validation, and chunking."""
    
    def __init__(self):
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    
    async def validate_audio_file(self, file_path: Path) -> Tuple[bool, str, Optional[float]]:
        """
        Validate audio file format, size, and duration.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (is_valid, error_message, duration)
        """
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > settings.MAX_FILE_SIZE:
                return False, f"File size {file_size} exceeds maximum allowed size {settings.MAX_FILE_SIZE}", None
            
            # Check file format
            if file_path.suffix.lower() not in self.supported_formats:
                return False, f"Unsupported file format: {file_path.suffix}", None
            
            # Get audio duration using ffprobe
            duration = self._get_audio_duration(file_path)
            if duration is None:
                return False, "Could not determine audio duration", None
            
            if duration > settings.MAX_AUDIO_DURATION:
                return False, f"Audio duration {duration}s exceeds maximum allowed duration {settings.MAX_AUDIO_DURATION}s", None
            
            return True, "", duration
            
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {str(e)}")
            return False, f"Error validating audio file: {str(e)}", None
    
    def _get_audio_duration(self, file_path: Path) -> Optional[float]:
        """Get audio duration using ffprobe."""
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return None
    
    async def convert_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """
        Convert audio file to WAV format with specified sample rate.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            logger.info(f"Converting {input_path} to WAV format")
            
            # Use ffmpeg to convert to WAV
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le',
                ac=1,  # Mono
                ar=settings.SAMPLE_RATE,  # Sample rate
                loglevel='error'
            )
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Successfully converted to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            return False
    
    async def chunk_audio(self, audio_path: Path, chunk_duration: int = None) -> List[Path]:
        """
        Split long audio files into smaller chunks for processing.
        
        Args:
            audio_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of paths to chunk files
        """
        if chunk_duration is None:
            chunk_duration = settings.CHUNK_DURATION
        
        try:
            duration = self._get_audio_duration(audio_path)
            if duration is None:
                raise ValueError("Could not determine audio duration")
            
            # If audio is shorter than chunk duration, return original file
            if duration <= chunk_duration:
                return [audio_path]
            
            logger.info(f"Chunking audio file of duration {duration}s into {chunk_duration}s chunks")
            
            chunks = []
            num_chunks = int(np.ceil(duration / chunk_duration))
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, duration)
                
                chunk_path = settings.TEMP_DIR / f"chunk_{i:04d}_{audio_path.stem}.wav"
                
                # Extract chunk using ffmpeg
                stream = ffmpeg.input(str(audio_path), ss=start_time, t=end_time - start_time)
                stream = ffmpeg.output(
                    stream,
                    str(chunk_path),
                    acodec='pcm_s16le',
                    ac=1,
                    ar=settings.SAMPLE_RATE,
                    loglevel='error'
                )
                
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                chunks.append(chunk_path)
                
                logger.debug(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            
            logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking audio file: {str(e)}")
            raise
    
    async def cleanup_temp_files(self, file_paths: List[Path]) -> None:
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if file_path.exists() and file_path.parent == settings.TEMP_DIR:
                    file_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file {file_path}: {str(e)}")
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def merge_srt_chunks(self, srt_chunks: List[str], chunk_duration: int = None) -> str:
        """
        Merge multiple SRT chunks into a single SRT file.
        
        Args:
            srt_chunks: List of SRT content strings
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Merged SRT content
        """
        if chunk_duration is None:
            chunk_duration = settings.CHUNK_DURATION
        
        merged_srt = []
        subtitle_index = 1
        
        for i, srt_chunk in enumerate(srt_chunks):
            if not srt_chunk.strip():
                continue
            
            # Parse SRT chunk and adjust timestamps
            lines = srt_chunk.strip().split('\n')
            time_offset = i * chunk_duration
            
            j = 0
            while j < len(lines):
                line = lines[j].strip()
                
                # Skip empty lines
                if not line:
                    j += 1
                    continue
                
                # Check if line is a number (subtitle index)
                if line.isdigit():
                    # Add new subtitle index
                    merged_srt.append(str(subtitle_index))
                    subtitle_index += 1
                    j += 1
                    
                    # Get timestamp line
                    if j < len(lines):
                        timestamp_line = lines[j].strip()
                        if ' --> ' in timestamp_line:
                            start_time, end_time = timestamp_line.split(' --> ')
                            
                            # Adjust timestamps
                            start_seconds = self._timestamp_to_seconds(start_time) + time_offset
                            end_seconds = self._timestamp_to_seconds(end_time) + time_offset
                            
                            new_timestamp = f"{self.format_timestamp(start_seconds)} --> {self.format_timestamp(end_seconds)}"
                            merged_srt.append(new_timestamp)
                            j += 1
                    
                    # Collect subtitle text
                    subtitle_text = []
                    while j < len(lines) and lines[j].strip():
                        subtitle_text.append(lines[j].strip())
                        j += 1
                    
                    if subtitle_text:
                        merged_srt.extend(subtitle_text)
                        merged_srt.append('')  # Empty line between subtitles
                else:
                    j += 1
        
        return '\n'.join(merged_srt)
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp to seconds."""
        time_parts = timestamp.replace(',', '.').split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        
        return hours * 3600 + minutes * 60 + seconds


# Global audio processor instance
audio_processor = AudioProcessor() 