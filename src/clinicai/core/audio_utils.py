"""
Audio processing utilities for normalization and duration extraction.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> str:
    """Get ffmpeg executable path from env or default."""
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def get_ffprobe_path() -> str:
    """Get ffprobe executable path from env or default."""
    return os.getenv("FFPROBE_PATH", "ffprobe")


def normalize_audio_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    timeout_seconds: int = 300,
) -> Tuple[str, float]:
    """
    Convert audio/video file to WAV 16kHz mono PCM format.

    Args:
        input_path: Path to input audio/video file
        output_path: Optional output path. If None, creates temp file.
        timeout_seconds: Maximum time for conversion (default: 5 minutes)

    Returns:
        Tuple of (output_path, conversion_time_seconds)

    Raises:
        FileNotFoundError: If ffmpeg is not found
        subprocess.CalledProcessError: If conversion fails
        TimeoutError: If conversion exceeds timeout
    """
    ffmpeg = get_ffmpeg_path()

    # Check if ffmpeg exists
    try:
        subprocess.run(
            [ffmpeg, "-version"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutError) as e:
        raise FileNotFoundError(f"ffmpeg not found or not executable. Set FFMPEG_PATH env var. Error: {e}")

    # Create output path if not provided
    if output_path is None:
        output_fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="normalized_")
        os.close(output_fd)

    conversion_start = os.times().elapsed

    try:
        # Convert to WAV 16kHz mono PCM
        # -i: input file
        # -ar 16000: sample rate 16kHz
        # -ac 1: mono (1 channel)
        # -f wav: WAV format
        # -y: overwrite output file
        subprocess.run(
            [
                ffmpeg,
                "-i",
                input_path,
                "-ar",
                "16000",  # Sample rate: 16kHz
                "-ac",
                "1",  # Channels: mono
                "-f",
                "wav",  # Format: WAV
                "-y",  # Overwrite output
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )

        conversion_time = time.time() - conversion_start

        logger.info(f"Audio normalized: {input_path} -> {output_path}, " f"conversion_time={conversion_time:.2f}s")

        return output_path, conversion_time

    except subprocess.TimeoutError:
        raise TimeoutError(f"Audio normalization timed out after {timeout_seconds}s for {input_path}")
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "Unknown error"
        raise RuntimeError(f"Audio normalization failed for {input_path}: {error_output[:500]}")


def get_audio_duration(input_path: str, timeout_seconds: int = 30) -> float:
    """
    Get audio/video duration in seconds using ffprobe.

    Args:
        input_path: Path to input audio/video file
        timeout_seconds: Maximum time for probe (default: 30 seconds)

    Returns:
        Duration in seconds

    Raises:
        FileNotFoundError: If ffprobe is not found
        subprocess.CalledProcessError: If probe fails
        TimeoutError: If probe exceeds timeout
    """
    ffprobe = get_ffprobe_path()

    # Check if ffprobe exists
    try:
        subprocess.run(
            [ffprobe, "-version"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutError) as e:
        raise FileNotFoundError(f"ffprobe not found or not executable. Set FFPROBE_PATH env var. Error: {e}")

    try:
        # Get duration using ffprobe
        # -v error: Only show errors
        # -show_entries format=duration: Show only duration
        # -of default=noprint_wrappers=1:nokey=1: Plain output
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_path,
            ],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
        )

        duration_str = result.stdout.strip()
        if not duration_str:
            raise ValueError(f"Could not extract duration from {input_path}")

        duration = float(duration_str)

        logger.debug(f"Audio duration: {input_path} -> {duration:.2f}s")

        return duration

    except subprocess.TimeoutError:
        raise TimeoutError(f"Audio duration probe timed out after {timeout_seconds}s for {input_path}")
    except (subprocess.CalledProcessError, ValueError) as e:
        error_output = e.stderr.decode("utf-8", errors="ignore") if hasattr(e, "stderr") and e.stderr else str(e)
        raise RuntimeError(f"Failed to get audio duration for {input_path}: {error_output[:500]}")
