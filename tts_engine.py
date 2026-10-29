from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

MODEL_ID = os.getenv("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")


@dataclass(frozen=True)
class SynthesisResult:
    sample_rate: int
    audio: np.ndarray
    elapsed_seconds: float
    device: str
    saved_path: Path


class IndicParlerEngine:
    """Lazy-loading, single-GPU Indic Parler-TTS inference engine."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.model = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None
        self.device = "not loaded"
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return

        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        token = (os.getenv("HF_TOKEN") or "").strip()
        if not token or token == "hf_your_read_token" or not token.startswith("hf_"):
            raise RuntimeError(
                "No valid Hugging Face token was loaded. Replace HF_TOKEN=hf_your_read_token "
                "in tts-research/.env with the real token from "
                "https://huggingface.co/settings/tokens, then restart the server."
            )
        requested_device = os.getenv("TTS_DEVICE", "auto").lower()
        use_cuda = torch.cuda.is_available() and requested_device != "cpu"
        target_device = torch.device("cuda:0" if use_cuda else "cpu")
        requested_dtype = os.getenv("TTS_DTYPE", "auto").lower()
        dtype_names = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if requested_dtype == "auto":
            # CUDA T4/P100 and newer: FP16. CPU: BF16 to halve model memory.
            dtype = torch.float16 if use_cuda else torch.bfloat16
        elif requested_dtype in dtype_names:
            dtype = dtype_names[requested_dtype]
        else:
            raise ValueError("TTS_DTYPE must be auto, float16, bfloat16, or float32")

        kwargs = {"token": token, "torch_dtype": dtype, "low_cpu_mem_usage": True}

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        # ParlerTTS does not implement Transformers' device_map='auto'. Move the
        # complete model explicitly instead of attempting GPU/CPU sharding.
        self.model.to(target_device)
        self.model.eval()

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token
        )
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path, token=token
        )
        self.device = f"{target_device} ({str(dtype).removeprefix('torch.')})"

    def synthesize(self, text: str, description: str) -> SynthesisResult:
        text, description = text.strip(), description.strip()
        if not text:
            raise ValueError("कृपया बोल्ने नेपाली पाठ लेख्नुहोस्।")
        if not description:
            raise ValueError("Voice description cannot be empty.")

        with self._lock:
            self.load()
            import torch

            # Inputs must start on the same device as the model's first parameter.
            input_device = next(self.model.parameters()).device
            desc = self.description_tokenizer(
                description, return_tensors="pt"
            ).to(input_device)
            prompt = self.prompt_tokenizer(text, return_tensors="pt").to(input_device)
            # Parler occasionally predicts EOS after the first Nepali word. One
            # audio-code token represents roughly 512/44100 seconds. Derive a
            # conservative floor from non-whitespace characters so it has time
            # to speak the complete sentence instead of returning a sub-second
            # clip. This can be overridden for experiments.
            spoken_chars = len(re.sub(r"\s+", "", text))
            min_new_tokens = int(
                os.getenv("TTS_MIN_NEW_TOKENS", str(max(128, min(1800, spoken_chars * 7))))
            )
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=desc.input_ids,
                    attention_mask=desc.attention_mask,
                    prompt_input_ids=prompt.input_ids,
                    prompt_attention_mask=prompt.attention_mask,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max(2048, min_new_tokens + 256),
                )
            audio = generated.detach().float().cpu().numpy().squeeze()
            elapsed = time.perf_counter() - started
            output_dir = Path(
                os.getenv("TTS_OUTPUT_DIR", Path(__file__).resolve().parent / "audios")
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            saved_path = output_dir / f"nepali-{stamp}.wav"
            import soundfile as sf

            # PCM_16 plays reliably in browsers, desktop players, and notebooks.
            sf.write(saved_path, audio, self.model.config.sampling_rate, subtype="PCM_16")
            return SynthesisResult(
                sample_rate=self.model.config.sampling_rate,
                audio=audio,
                elapsed_seconds=elapsed,
                device=self.device,
                saved_path=saved_path,
            )
