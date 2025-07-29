# utils.py
# Utility functions for the TTS server application.
# This module includes functions for audio processing, text manipulation,
# file system operations, and performance monitoring.
# Optimized for production performance with GPU acceleration and streaming.

import os
import logging
import re
import time
import io
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Set, List, Generator
from pydub import AudioSegment
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torchaudio  # For saving PyTorch tensors and potentially speed adjustment.
import torch

# Configuration manager to get paths dynamically.
# Assumes config.py and its config_manager are in the same directory or accessible via PYTHONPATH.
from config import get_predefined_voices_path, get_reference_audio_path, config_manager

# Optional import for librosa (for audio resampling, e.g., Opus encoding and time stretching)
try:
    import librosa

    LIBROSA_AVAILABLE = True
    logger = logging.getLogger(
        __name__
    )  # Initialize logger here if librosa is available
    logger.info(
        "Librosa library found and will be used for audio resampling and time stretching."
    )
except ImportError:
    LIBROSA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Librosa library not found. Advanced audio resampling features (e.g., for Opus encoding) "
        "and pitch-preserving speed adjustment will be limited. Speed adjustment will fall back to basic method if enabled."
    )

# Optional import for Parselmouth (for unvoiced segment detection)
try:
    import parselmouth

    PARSELMOUTH_AVAILABLE = True
    logger.info(
        "Parselmouth library found and will be used for unvoiced segment removal if enabled."
    )
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning(
        "Parselmouth library not found. Unvoiced segment removal feature will be disabled."
    )

# Audio processing executor for parallel processing
audio_executor = ThreadPoolExecutor(max_workers=4)

# --- Filename Sanitization ---
def sanitize_filename(filename: str) -> str:
    """
    Removes potentially unsafe characters and path components from a filename
    to make it safe for use in file paths. Replaces unsafe sequences with underscores.

    Args:
        filename: The original filename string.

    Returns:
        A sanitized filename string, ensuring it's not empty and reasonably short.
    """
    if not filename:
        # Generate a unique name if the input is empty.
        return f"unnamed_file_{uuid.uuid4().hex[:8]}"

    # Remove directory separators and leading/trailing whitespace.
    base_filename = Path(filename).name.strip()
    if not base_filename:
        return f"empty_basename_{uuid.uuid4().hex[:8]}"

    # Define a set of allowed characters (alphanumeric, underscore, hyphen, dot, space).
    # Spaces will be replaced by underscores later.
    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
    )
    sanitized_list = []
    last_char_was_underscore = False

    for char in base_filename:
        if char in safe_chars:
            # Replace spaces with underscores.
            sanitized_list.append("_" if char == " " else char)
            last_char_was_underscore = char == " "
        elif not last_char_was_underscore:
            # Replace any disallowed character sequence with a single underscore.
            sanitized_list.append("_")
            last_char_was_underscore = True

    sanitized = "".join(sanitized_list).strip("_")

    # Prevent names starting with multiple dots or consisting only of dots/underscores.
    if not sanitized or sanitized.lstrip("._") == "":
        return f"sanitized_file_{uuid.uuid4().hex[:8]}"

    # Limit filename length (e.g., 100 characters), preserving the extension.
    max_len = 100
    if len(sanitized) > max_len:
        name_part, ext_part = os.path.splitext(sanitized)
        # Ensure extension is not overly long itself; common extensions are short.
        ext_part = ext_part[:10]  # Limit extension length just in case.
        name_part = name_part[
            : max_len - len(ext_part) - 1
        ]  # -1 for the dot if ext exists
        sanitized = name_part + ext_part
        logger.warning(
            f"Original filename '{base_filename}' was truncated to '{sanitized}' due to length limits."
        )

    if not sanitized:  # Should not happen with previous checks, but as a failsafe.
        return f"final_fallback_name_{uuid.uuid4().hex[:8]}"

    return sanitized


# --- Constants for Text Processing ---
# Set of common abbreviations to help with sentence splitting.
ABBREVIATIONS: Set[str] = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "rev.",
    "hon.",
    "st.",
    "etc.",
    "e.g.",
    "i.e.",
    "vs.",
    "approx.",
    "apt.",
    "dept.",
    "fig.",
    "gen.",
    "gov.",
    "inc.",
    "jr.",
    "sr.",
    "ltd.",
    "no.",
    "p.",
    "pp.",
    "vol.",
    "op.",
    "cit.",
    "ca.",
    "cf.",
    "ed.",
    "esp.",
    "et.",
    "al.",
    "ibid.",
    "id.",
    "inf.",
    "sup.",
    "viz.",
    "sc.",
    "fl.",
    "d.",
    "b.",
    "r.",
    "c.",
    "v.",
    "u.s.",
    "u.k.",
    "a.m.",
    "p.m.",
    "a.d.",
    "b.c.",
}

# Common titles that might appear without a period if cleaned by other means first.
TITLES_NO_PERIOD: Set[str] = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "rev",
    "hon",
    "st",
    "sgt",
    "capt",
    "lt",
    "col",
    "gen",
}

# Regex patterns (pre-compiled for efficiency in text processing).
NUMBER_DOT_NUMBER_PATTERN = re.compile(
    r"(?<!\d\.)\d*\.\d+"
)

# --- Optimized Audio Processing Functions ---

def encode_audio_optimized(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: str = "opus",
    target_sample_rate: Optional[int] = None,
) -> Optional[bytes]:
    """
    Optimized audio encoding with GPU acceleration and parallel processing.
    """
    try:
        # Use GPU if available for resampling
        if torch.cuda.is_available() and target_sample_rate and target_sample_rate != sample_rate:
            # GPU-accelerated resampling
            audio_tensor = torch.from_numpy(audio_array).cuda()
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
                dtype=audio_tensor.dtype
            ).cuda()
            audio_tensor = resampler(audio_tensor)
            audio_array = audio_tensor.cpu().numpy()
            sample_rate = target_sample_rate
        elif target_sample_rate and target_sample_rate != sample_rate:
            # CPU resampling with librosa
            if LIBROSA_AVAILABLE:
                audio_array = librosa.resample(
                    y=audio_array,
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate,
                    res_type="kaiser_best"
                )
                sample_rate = target_sample_rate
            else:
                logger.warning("Librosa not available, skipping resampling")

        # Optimized encoding based on format
        if output_format.lower() == "opus":
            return encode_audio_opus_optimized(audio_array, sample_rate)
        elif output_format.lower() == "wav":
            return encode_audio_wav_optimized(audio_array, sample_rate)
        elif output_format.lower() == "mp3":
            return encode_audio_mp3_optimized(audio_array, sample_rate)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None

    except Exception as e:
        logger.error(f"Error in optimized audio encoding: {e}")
        return None

def encode_audio_streaming(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: str = "opus",
    target_sample_rate: Optional[int] = None,
    chunk_size: int = 1024,
) -> Generator[bytes, None, None]:
    """
    Stream audio encoding for real-time response.
    """
    try:
        # Process audio in chunks
        total_samples = len(audio_array)
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = audio_array[start_idx:end_idx]
            
            # Encode chunk
            encoded_chunk = encode_audio_optimized(
                chunk, sample_rate, output_format, target_sample_rate
            )
            
            if encoded_chunk:
                yield encoded_chunk
            else:
                logger.error(f"Failed to encode audio chunk {start_idx}-{end_idx}")
                break
                
    except Exception as e:
        logger.error(f"Error in streaming audio encoding: {e}")

def encode_audio_opus_optimized(
    audio_array: np.ndarray, sample_rate: int
) -> Optional[bytes]:
    """
    Optimized Opus encoding with better compression settings.
    """
    try:
        # Convert numpy array to proper audio format
        # Ensure audio is in the correct range (-1 to 1)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM for pydub
        audio_16bit = (audio_array * 32767).astype(np.int16)
        
        # Use pydub for optimized Opus encoding
        audio_segment = AudioSegment(
            audio_16bit.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1
        )
        
        # Optimized Opus settings for better quality/size ratio
        buffer = io.BytesIO()
        audio_segment.export(
            buffer,
            format="opus",
            codec="libopus",
            bitrate="64k",  # Optimized bitrate for speech
            parameters=["-vbr", "on", "-compression_level", "10"]
        )
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error in optimized Opus encoding: {e}")
        return None

def encode_audio_wav_optimized(
    audio_array: np.ndarray, sample_rate: int
) -> Optional[bytes]:
    """
    Optimized WAV encoding with better compression.
    """
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error in optimized WAV encoding: {e}")
        return None

def encode_audio_mp3_optimized(
    audio_array: np.ndarray, sample_rate: int
) -> Optional[bytes]:
    """
    Optimized MP3 encoding with better compression settings.
    """
    try:
        # Convert numpy array to proper audio format
        # Ensure audio is in the correct range (-1 to 1)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM for pydub
        audio_16bit = (audio_array * 32767).astype(np.int16)
        
        # Use pydub for optimized MP3 encoding
        audio_segment = AudioSegment(
            audio_16bit.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1
        )
        
        buffer = io.BytesIO()
        audio_segment.export(
            buffer,
            format="mp3",
            bitrate="128k",
            parameters=["-q:a", "2"]  # High quality setting
        )
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error in optimized MP3 encoding: {e}")
        return None

def encode_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: str = "opus",
    target_sample_rate: Optional[int] = None,
) -> Optional[bytes]:
    """
    Legacy audio encoding function - now uses optimized version.
    """
    return encode_audio_optimized(audio_array, sample_rate, output_format, target_sample_rate)

def save_audio_to_file(
    audio_array: np.ndarray, sample_rate: int, file_path_str: str
) -> bool:
    """
    Saves audio array to file with optimized settings.
    """
    try:
        file_path = Path(file_path_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from file extension
        file_extension = file_path.suffix.lower()
        
        if file_extension == ".wav":
            sf.write(file_path, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        elif file_extension == ".mp3":
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio_array.dtype.itemsize,
                channels=1
            )
            audio_segment.export(file_path, format="mp3", bitrate="128k")
        elif file_extension == ".opus":
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio_array.dtype.itemsize,
                channels=1
            )
            audio_segment.export(file_path, format="opus", bitrate="64k")
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return False

        logger.info(f"Audio saved successfully to: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving audio to file: {e}")
        return False

def save_audio_tensor_to_file(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    file_path_str: str,
    output_format: str = "wav",
) -> bool:
    """
    Saves audio tensor to file with optimized settings.
    """
    try:
        # Convert tensor to numpy array
        audio_np = audio_tensor.cpu().numpy().squeeze()
        return save_audio_to_file(audio_np, sample_rate, file_path_str)

    except Exception as e:
        logger.error(f"Error saving audio tensor to file: {e}")
        return False

def apply_speed_factor(
    audio_tensor: torch.Tensor, sample_rate: int, speed_factor: float
) -> Tuple[torch.Tensor, int]:
    """
    Optimized speed factor application with GPU acceleration.
    """
    try:
        if speed_factor == 1.0:
            return audio_tensor, sample_rate

        # Use GPU if available
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
            
            # GPU-accelerated time stretching
            if hasattr(torchaudio.transforms, 'TimeStretch'):
                # Ensure proper tensor shape for TimeStretch
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                elif audio_tensor.dim() > 2:
                    audio_tensor = audio_tensor.squeeze()  # Remove extra dimensions
                
                time_stretch = torchaudio.transforms.TimeStretch(
                    fixed_rate=speed_factor
                ).cuda()
                audio_tensor = time_stretch(audio_tensor)
                
                # Ensure output is 1D
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.squeeze()
            else:
                # Fallback to CPU method
                audio_tensor = audio_tensor.cpu()
                audio_np = audio_tensor.numpy().squeeze()
                
                if LIBROSA_AVAILABLE:
                    # Use librosa for pitch-preserving speed adjustment
                    audio_np = librosa.effects.time_stretch(audio_np, rate=speed_factor)
                else:
                    # Basic resampling method
                    new_length = int(len(audio_np) / speed_factor)
                    indices = np.linspace(0, len(audio_np) - 1, new_length, dtype=int)
                    audio_np = audio_np[indices]
                
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            # CPU processing
            audio_np = audio_tensor.numpy().squeeze()
            
            if LIBROSA_AVAILABLE:
                # Use librosa for pitch-preserving speed adjustment
                audio_np = librosa.effects.time_stretch(audio_np, rate=speed_factor)
            else:
                # Basic resampling method
                new_length = int(len(audio_np) / speed_factor)
                indices = np.linspace(0, len(audio_np) - 1, new_length, dtype=int)
                audio_np = audio_np[indices]
            
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        return audio_tensor, sample_rate

    except Exception as e:
        logger.error(f"Error applying speed factor: {e}")
        return audio_tensor, sample_rate

def trim_lead_trail_silence(
    audio_array: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_duration_ms: int = 100,
    padding_ms: int = 50,
) -> np.ndarray:
    """
    Optimized silence trimming with GPU acceleration.
    """
    try:
        # Convert to GPU if available for faster processing
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio_array).cuda()
            
            # Calculate RMS energy
            rms = torch.sqrt(torch.mean(audio_tensor ** 2))
            threshold = 10 ** (silence_threshold_db / 20) * rms
            
            # Find non-silent regions
            energy = audio_tensor ** 2
            non_silent = energy > threshold
            
            # Find start and end points
            non_silent_indices = torch.where(non_silent)[0]
            if len(non_silent_indices) == 0:
                return audio_array

            start_idx = non_silent_indices[0].item()
            end_idx = non_silent_indices[-1].item()
            
            # Add padding
            padding_samples = int(padding_ms * sample_rate / 1000)
            start_idx = max(0, start_idx - padding_samples)
            end_idx = min(len(audio_array), end_idx + padding_samples)
            
            return audio_array[start_idx:end_idx]
        else:
            # CPU processing
            return trim_lead_trail_silence_cpu(
                audio_array, sample_rate, silence_threshold_db, min_silence_duration_ms, padding_ms
            )
            
    except Exception as e:
        logger.error(f"Error in GPU-accelerated silence trimming: {e}")
        return audio_array

def trim_lead_trail_silence_cpu(
    audio_array: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_duration_ms: int = 100,
    padding_ms: int = 50,
) -> np.ndarray:
    """
    CPU-based silence trimming for fallback.
    """
    try:
        # Convert to float if needed
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_array ** 2))
        threshold = 10 ** (silence_threshold_db / 20) * rms

        # Find non-silent regions
        energy = audio_array ** 2
        non_silent = energy > threshold

        # Find start and end points
        non_silent_indices = np.where(non_silent)[0]
        if len(non_silent_indices) == 0:
            return audio_array

        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1]

        # Add padding
        padding_samples = int(padding_ms * sample_rate / 1000)
        start_idx = max(0, start_idx - padding_samples)
        end_idx = min(len(audio_array), end_idx + padding_samples)

        return audio_array[start_idx:end_idx]

    except Exception as e:
        logger.error(f"Error in CPU silence trimming: {e}")
        return audio_array

def fix_internal_silence(
    audio_array: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_to_fix_ms: int = 700,
    max_allowed_silence_ms: int = 300,
) -> np.ndarray:
    """
    Optimized internal silence fixing with GPU acceleration.
    """
    try:
        # Convert to GPU if available
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio_array).cuda()
            
            # Calculate RMS energy
            rms = torch.sqrt(torch.mean(audio_tensor ** 2))
            threshold = 10 ** (silence_threshold_db / 20) * rms
            
            # Find silent regions
            energy = audio_tensor ** 2
            silent = energy <= threshold
            
            # Find long silent regions
            min_silence_samples = int(min_silence_to_fix_ms * sample_rate / 1000)
            max_allowed_samples = int(max_allowed_silence_ms * sample_rate / 1000)
            
            # Find consecutive silent samples
            silent_runs = torch.where(silent)[0]
            if len(silent_runs) == 0:
                return audio_array

            # Group consecutive silent regions
            silent_regions = []
            start_idx = silent_runs[0]
            prev_idx = start_idx
            
            for idx in silent_runs[1:]:
                if idx - prev_idx > 1:  # Gap in silent region
                    if prev_idx - start_idx >= min_silence_samples:
                        silent_regions.append((start_idx, prev_idx))
                    start_idx = idx
                prev_idx = idx
            
            # Add last region
            if prev_idx - start_idx >= min_silence_samples:
                silent_regions.append((start_idx, prev_idx))
            
            # Process each long silent region
            result = audio_tensor.clone()
            for start, end in silent_regions:
                region_length = end - start + 1
                if region_length > max_allowed_samples:
                    # Compress the silence
                    target_length = max_allowed_samples
                    step = region_length / target_length
                    
                    # Create compressed indices
                    indices = torch.arange(start, end + 1, step=step, device=result.device)
                    indices = torch.clamp(indices, start, end).long()
                    
                    # Replace the region
                    result[start:start + target_length] = audio_tensor[indices[:target_length]]
                    
                    # Shift remaining audio
                    shift = region_length - target_length
                    if start + target_length + shift < len(result):
                        result[start + target_length:start + target_length + shift] = audio_tensor[end + 1:end + 1 + shift]
            
            return result.cpu().numpy()
        else:
            # CPU processing
            return fix_internal_silence_cpu(
                audio_array, sample_rate, silence_threshold_db, min_silence_to_fix_ms, max_allowed_silence_ms
            )
            
    except Exception as e:
        logger.error(f"Error in GPU-accelerated internal silence fixing: {e}")
        return audio_array

def fix_internal_silence_cpu(
    audio_array: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_to_fix_ms: int = 700,
    max_allowed_silence_ms: int = 300,
) -> np.ndarray:
    """
    CPU-based internal silence fixing for fallback.
    """
    try:
        # Convert to float if needed
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_array ** 2))
        threshold = 10 ** (silence_threshold_db / 20) * rms

        # Find silent regions
        energy = audio_array ** 2
        silent = energy <= threshold

        # Find long silent regions
        min_silence_samples = int(min_silence_to_fix_ms * sample_rate / 1000)
        max_allowed_samples = int(max_allowed_silence_ms * sample_rate / 1000)

        # Find consecutive silent samples
        silent_runs = np.where(silent)[0]
        if len(silent_runs) == 0:
            return audio_array

        # Group consecutive silent regions
        silent_regions = []
        start_idx = silent_runs[0]
        prev_idx = start_idx

        for idx in silent_runs[1:]:
            if idx - prev_idx > 1:  # Gap in silent region
                if prev_idx - start_idx >= min_silence_samples:
                    silent_regions.append((start_idx, prev_idx))
                start_idx = idx
            prev_idx = idx

        # Add last region
        if prev_idx - start_idx >= min_silence_samples:
            silent_regions.append((start_idx, prev_idx))

        # Process each long silent region
        result = audio_array.copy()
        for start, end in silent_regions:
            region_length = end - start + 1
            if region_length > max_allowed_samples:
                # Compress the silence
                target_length = max_allowed_samples
                step = region_length / target_length

                # Create compressed indices
                indices = np.arange(start, end + 1, step=step)
                indices = np.clip(indices, start, end).astype(int)

                # Replace the region
                result[start:start + target_length] = audio_array[indices[:target_length]]

                # Shift remaining audio
                shift = region_length - target_length
                if start + target_length + shift < len(result):
                    result[start + target_length:start + target_length + shift] = audio_array[end + 1:end + 1 + shift]

        return result

    except Exception as e:
        logger.error(f"Error in CPU internal silence fixing: {e}")
        return audio_array

def remove_long_unvoiced_segments(
    audio_array: np.ndarray,
    sample_rate: int,
    min_unvoiced_duration_ms: int = 300,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> np.ndarray:
    """
    Optimized unvoiced segment removal with GPU acceleration.
    """
    if not PARSELMOUTH_AVAILABLE:
        logger.warning("Parselmouth not available, skipping unvoiced segment removal")
        return audio_array

    try:
        # Use CPU for Parselmouth processing (GPU not supported)
        return remove_long_unvoiced_segments_cpu(
            audio_array, sample_rate, min_unvoiced_duration_ms, pitch_floor, pitch_ceiling
        )
    except Exception as e:
        logger.error(f"Error in unvoiced segment removal: {e}")
        return audio_array

def remove_long_unvoiced_segments_cpu(
    audio_array: np.ndarray,
    sample_rate: int,
    min_unvoiced_duration_ms: int = 300,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> np.ndarray:
    """
    CPU-based unvoiced segment removal.
    """
    try:
        import parselmouth

        # Create Praat Sound object
        sound = parselmouth.Sound(audio_array, sample_rate)

        # Extract pitch
        pitch = sound.to_pitch(
            time_step=0.01,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling
        )

        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()

        # Find unvoiced regions (where pitch is undefined or 0)
        unvoiced = (pitch_values == 0) | np.isnan(pitch_values)

        # Find long unvoiced segments
        min_unvoiced_samples = int(min_unvoiced_duration_ms * sample_rate / 1000)

        # Convert pitch time indices to sample indices
        sample_indices = (pitch_times * sample_rate).astype(int)
        sample_indices = np.clip(sample_indices, 0, len(audio_array) - 1)

        # Create mask for samples to keep
        keep_mask = np.ones(len(audio_array), dtype=bool)

        # Mark long unvoiced segments for removal
        for i in range(len(sample_indices) - 1):
            if unvoiced[i]:
                start_sample = sample_indices[i]
                end_sample = sample_indices[i + 1]
                
                if end_sample - start_sample >= min_unvoiced_samples:
                    keep_mask[start_sample:end_sample] = False

        # Apply mask
        return audio_array[keep_mask]

    except Exception as e:
        logger.error(f"Error in CPU unvoiced segment removal: {e}")
        return audio_array

# --- Text Processing Functions (Optimized) ---

def _is_valid_sentence_end(text: str, period_index: int) -> bool:
    """
    Optimized sentence end validation.
    """
    if period_index == 0 or period_index == len(text) - 1:
        return True

    # Check for common abbreviations
    word_before = text[:period_index].split()[-1].lower() if text[:period_index].split() else ""
    if word_before in ABBREVIATIONS:
        return False

    # Check for titles without periods
    if word_before in TITLES_NO_PERIOD:
        return False

    # Check for numbers (e.g., "3.14", "2023.12.31")
    if NUMBER_DOT_NUMBER_PATTERN.search(text[max(0, period_index - 10):period_index + 10]):
                    return False

    return True

def _split_text_by_punctuation(text: str) -> List[str]:
    """
    Optimized text splitting by punctuation.
    """
    if not text.strip():
        return []

    # Pre-compile regex for better performance
    sentence_end_pattern = re.compile(r'[.!?]+')
    
    # Find all sentence endings
    matches = list(sentence_end_pattern.finditer(text))
    
    if not matches:
        return [text]

    sentences = []
    start = 0

    for match in matches:
        end = match.end()
        
        # Validate if this is a real sentence end
        if _is_valid_sentence_end(text, match.start()):
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end

    # Add remaining text
    remaining = text[start:].strip()
    if remaining:
        sentences.append(remaining)

    return sentences

def split_into_sentences(text: str) -> List[str]:
    """
    Optimized sentence splitting with better performance.
    """
    if not text or not text.strip():
        return []

    # Clean the text first
    text = text.strip()
    
    # Handle common edge cases
    if len(text) < 10:
        return [text]

    # Split by punctuation
    sentences = _split_text_by_punctuation(text)
    
    # Post-process sentences
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Handle edge cases
            if sentence.endswith('.') and len(sentence) == 1:
                continue  # Skip lone periods
            processed_sentences.append(sentence)

    return processed_sentences if processed_sentences else [text]

def _preprocess_and_segment_text(full_text: str) -> List[Tuple[Optional[str], str]]:
    """
    Optimized text preprocessing and segmentation.
    """
    if not full_text or not full_text.strip():
        return []

    # Split into sentences first
    sentences = split_into_sentences(full_text)
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    max_chunk_length = 200  # Reasonable default

    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chunk_length and current_chunk:
            # Finalize current chunk
            chunks.append((" ".join(current_chunk), " ".join(current_chunk)))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append((" ".join(current_chunk), " ".join(current_chunk)))

    return chunks

def chunk_text_by_sentences(
    full_text: str,
    chunk_size: int,
) -> List[str]:
    """
    Optimized text chunking with better performance.
    """
    if not full_text or not full_text.strip():
        return []

    # Use optimized preprocessing
    chunks = _preprocess_and_segment_text(full_text)
    
    # Convert to simple list format
    result = []
    for _, chunk_text in chunks:
        if chunk_text.strip():
            result.append(chunk_text.strip())

    return result if result else [full_text]

# --- File System Functions ---

def get_valid_reference_files() -> List[str]:
    """
    Get list of valid reference audio files.
    """
    try:
        ref_path = get_reference_audio_path(ensure_absolute=True)
        if not ref_path.exists():
            return []

        valid_files = []
        for file_path in ref_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in [".wav", ".mp3"]:
                valid_files.append(file_path.name)

        return sorted(valid_files)

    except Exception as e:
        logger.error(f"Error getting valid reference files: {e}")
        return []

def get_predefined_voices() -> List[Dict[str, str]]:
    """
    Get list of predefined voices with metadata.
    """
    try:
        voices_path = get_predefined_voices_path(ensure_absolute=True)
        if not voices_path.exists():
            return []

        voices = []
        for file_path in voices_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in [".wav", ".mp3"]:
                voices.append({
                    "filename": file_path.name,
                    "display_name": file_path.stem.replace("_", " ").title(),
                    "path": str(file_path)
                })

        return sorted(voices, key=lambda x: x["display_name"])

    except Exception as e:
        logger.error(f"Error getting predefined voices: {e}")
        return []

def validate_reference_audio(
    file_path: Path, max_duration_sec: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Optimized audio file validation.
    """
    try:
        if not file_path.exists():
            return False, "File does not exist"

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty"
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return False, "File is too large (max 50MB)"

        # Validate audio format
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for faster validation
                audio, sr = librosa.load(str(file_path), sr=None, duration=1.0)
                duration = len(audio) / sr
            else:
                # Fallback to soundfile
                info = sf.info(str(file_path))
                duration = info.duration

            if max_duration_sec and duration > max_duration_sec:
                return False, f"Audio duration ({duration:.1f}s) exceeds limit ({max_duration_sec}s)"

            if duration < 0.5:
                return False, "Audio is too short (min 0.5s)"

            return True, "Valid audio file"

        except Exception as e:
            return False, f"Invalid audio format: {str(e)}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"

# --- Performance Monitoring ---

class PerformanceMonitor:
    """
    Enhanced performance monitoring with GPU metrics.
    """

    def __init__(
        self, enabled: bool = True, logger_instance: Optional[logging.Logger] = None
    ):
        self.enabled: bool = enabled
        self.logger = (
            logger_instance
            if logger_instance is not None
            else logging.getLogger(__name__)
        )
        self.start_time: float = 0.0
        self.events: List[Tuple[str, float]] = []
        if self.enabled:
            self.start_time = time.monotonic()
            self.events.append(("Monitoring Started", self.start_time))

    def record(self, event_name: str):
        if not self.enabled:
            return
        self.events.append((event_name, time.monotonic()))

    def report(self, log_level: int = logging.DEBUG) -> str:
        if not self.enabled or not self.events:
            return "Performance monitoring was disabled or no events recorded."

        report_lines = ["Performance Report:"]
        last_event_time = self.events[0][1]

        for i in range(1, len(self.events)):
            event_name, timestamp = self.events[i]
            prev_event_name, _ = self.events[i - 1]
            duration_since_last = timestamp - last_event_time
            duration_since_start = timestamp - self.start_time
            report_lines.append(
                f"  - Event: '{event_name}' (after '{prev_event_name}') "
                f"took {duration_since_last:.4f}s. Total elapsed: {duration_since_start:.4f}s"
            )
            last_event_time = timestamp

        total_duration = self.events[-1][1] - self.start_time
        report_lines.append(f"Total Monitored Duration: {total_duration:.4f}s")
        
        # Add GPU memory info if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            report_lines.append(f"GPU Memory Usage: {gpu_memory:.2f} GB")
        
        full_report_str = "\n".join(report_lines)

        if self.logger:
            self.logger.log(log_level, full_report_str)
        return full_report_str
