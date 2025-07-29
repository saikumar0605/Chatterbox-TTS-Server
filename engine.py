# File: engine.py
# Core TTS model loading and speech generation logic.
# Optimized for production performance with quantization and memory management.

import logging
import random
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import gc
import time
from contextlib import contextmanager

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

# Patch for chatterbox/models/s3gen/utils/mel.py to fix KeyError: 'cuda:0' on concurrent requests
import torch

def patch_hann_window():
    try:
        from chatterbox.models.s3gen.utils import mel
        if not hasattr(mel, "hann_window"):
            return
        def get_hann_window(device):
            if str(device) not in mel.hann_window:
                mel.hann_window[str(device)] = torch.hann_window(
                    mel.N_FFT, device=device
                )
            return mel.hann_window[str(device)]
        mel.get_hann_window = get_hann_window
        # Patch the mel_spectrogram function to always call get_hann_window
        orig_mel_spectrogram = mel.mel_spectrogram
        def patched_mel_spectrogram(y, sr, n_fft=None, hop_length=None, win_length=None, window=None, center=True, pad_mode="reflect", power=2.0, device=None):
            device = y.device if hasattr(y, 'device') else torch.device("cpu")
            mel.get_hann_window(device)
            return orig_mel_spectrogram(y, sr, n_fft, hop_length, win_length, window, center, pad_mode, power, device)
        mel.mel_spectrogram = patched_mel_spectrogram
    except Exception as e:
        print(f"[WARN] Failed to patch hann_window for chatterbox: {e}")

patch_hann_window()

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = None
model_dtype: torch.dtype = torch.float16  # Use FP16 for better performance
is_quantized: bool = False
model_warmup_complete: bool = False

# Performance monitoring
inference_times: list = []
memory_usage: Dict[str, float] = {}

@contextmanager
def torch_gc_context():
    """Context manager for automatic GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def optimize_model_for_inference(model: ChatterboxTTS) -> ChatterboxTTS:
    """
    Optimize the model for faster inference with quantization and memory optimization.
    """
    global model_dtype, is_quantized
    
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Move model to specified device and dtype
        if model_device == "cuda":
            model = model.to(device=model_device, dtype=model_dtype)
            
            # Enable TensorFloat-32 for faster computation on Ampere+ GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Optimize CUDA memory allocation
            torch.cuda.empty_cache()
            
            # Quantize model for faster inference (INT8 quantization)
            try:
                if hasattr(model, 'quantize'):
                    model = model.quantize()
                    is_quantized = True
                    logger.info("Model quantized successfully for faster inference")
                else:
                    # Manual quantization for better performance
                    for module in model.modules():
                        if hasattr(module, 'weight') and module.weight is not None:
                            if module.weight.dtype == torch.float32:
                                module.weight.data = module.weight.data.half()
                    is_quantized = True
                    logger.info("Model manually quantized to FP16")
            except Exception as e:
                logger.warning(f"Quantization failed, using FP16: {e}")
                is_quantized = False
        
        # Enable JIT compilation for faster inference
        try:
            if hasattr(model, 'jit_compile'):
                model = model.jit_compile()
                logger.info("Model JIT compiled for faster inference")
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
        
        logger.info(f"Model optimized for inference on {model_device} with {model_dtype}")
        return model
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return model

def warmup_model(model: ChatterboxTTS) -> bool:
    """
    Warm up the model with dummy inference to optimize CUDA kernels and memory allocation.
    """
    global model_warmup_complete
    
    try:
        logger.info("Starting model warmup...")
        
        # Warmup with different text lengths to optimize various kernel paths
        warmup_texts = [
            "Hello world.",
            "This is a longer warmup text to optimize the model for various input lengths.",
            "Short.",
            "A medium length text for warmup purposes."
        ]
        
        for i, text in enumerate(warmup_texts):
            try:
                with torch.no_grad():
                    with torch_gc_context():
                        start_time = time.time()
                        result = model.generate(
                            text=text,
                            audio_prompt_path=None,
                            temperature=0.5,
                            exaggeration=0.5,
                            cfg_weight=0.5,
                        )
                        inference_time = time.time() - start_time
                        
                        # Handle different return formats
                        if isinstance(result, tuple):
                            _audio, _sr = result
                        else:
                            _audio = result
                            _sr = model.sr  # Use model's sample rate
                        logger.info(f"Warmup {i+1}/{len(warmup_texts)} completed in {inference_time:.3f}s")
                        
                        # Record performance metrics
                        inference_times.append(inference_time)
                        
                        if torch.cuda.is_available():
                            memory_usage[f"warmup_{i+1}"] = torch.cuda.memory_allocated() / 1024**3
                            
            except Exception as e:
                logger.warning(f"Warmup {i+1} failed: {e}")
                continue
        
        model_warmup_complete = True
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        logger.info(f"Model warmup complete. Average inference time: {avg_inference_time:.3f}s")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        return False

def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")

def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False

def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False

def load_model() -> bool:
    """
    Loads and optimizes the TTS model for production inference.
    This version includes quantization, memory optimization, and warmup.
    """
    global chatterbox_model, MODEL_LOADED, model_device, model_dtype

    if MODEL_LOADED:
        logger.info("TTS model is already loaded and optimized.")
        return True

    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        
        # Set dtype based on device
        if model_device == "cuda":
            model_dtype = torch.float16  # Use FP16 for better performance on GPU
        else:
            model_dtype = torch.float32  # Use FP32 for CPU/MPS for better accuracy
        
        logger.info(f"Final device selection: {model_device} with dtype: {model_dtype}")

        # Get configured model_repo_id for logging and context
        model_repo_id_config = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )

        logger.info(
            f"Loading model with optimization for production inference..."
        )
        
        try:
            # Load model with optimization flags
            chatterbox_model = ChatterboxTTS.from_pretrained(
                device=model_device,
            )
            
            # Optimize model for inference
            chatterbox_model = optimize_model_for_inference(chatterbox_model)
            
            # Warm up the model
            if not warmup_model(chatterbox_model):
                logger.warning("Model warmup failed, but continuing with inference")
            
            logger.info(
                f"Successfully loaded and optimized TTS model on {model_device} "
                f"(expected from '{model_repo_id_config}' or library default)."
            )
            
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained: {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded and optimized successfully on {model_device}. "
                f"Engine sample rate: {chatterbox_model.sr} Hz. "
                f"Quantized: {is_quantized}"
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. "
                "This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False

def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the optimized TTS model.
    Includes performance monitoring and memory management.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    start_time = time.time()
    
    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}"
        )

        # Use context manager for automatic memory cleanup
        with torch_gc_context():
            with torch.no_grad():  # Disable gradients for inference
                # Call the core model's generate method
                result = chatterbox_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
                
                # Handle different return formats
                if isinstance(result, tuple):
                    wav_tensor, sr = result
                else:
                    wav_tensor = result
                    sr = chatterbox_model.sr

        # Record performance metrics
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            memory_usage[f"inference_{len(inference_times)}"] = current_memory
        
        logger.debug(f"Inference completed in {inference_time:.3f}s")
        
        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None

def get_performance_stats() -> Dict[str, Any]:
    """
    Get performance statistics for monitoring.
    """
    stats = {
        "model_loaded": MODEL_LOADED,
        "device": model_device,
        "quantized": is_quantized,
        "warmup_complete": model_warmup_complete,
        "dtype": str(model_dtype),
    }
    
    if inference_times:
        stats.update({
            "avg_inference_time": sum(inference_times) / len(inference_times),
            "min_inference_time": min(inference_times),
            "max_inference_time": max(inference_times),
            "total_inferences": len(inference_times),
        })
    
    if torch.cuda.is_available():
        stats.update({
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            "gpu_memory_peak": torch.cuda.max_memory_allocated() / 1024**3,
        })
    
    return stats

# --- End File: engine.py ---
