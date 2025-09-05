import asyncio
import logging
import os
import gc
from typing import Dict, Any, Tuple, Optional

import torch
import whisperx
from pydantic import BaseModel

# ──────────────────────────────  Logging  ──────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("configurable_transcription")


# ────────────────────────  Public data structures  ─────────────────────
class TranscriptionConfig(BaseModel):
    """Configuration parameters for WhisperX transcription."""
    # Basic
    language: Optional[str] = "de"
    compute_type: str = "float16"          # Default to float16 for better GPU performance
    model_name: str = "large-v3"

    # Transcription
    temperature: float = 0.0
    beam_size: int = 5
    word_timestamps: bool = True
    batch_size: int = 16
    condition_on_previous_text: bool = True

    # Silence / VAD
    vad_filter: bool = True
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    vad_onset: float = 0.500
    vad_offset: float = 0.363

    # Alignment
    align_output: bool = True


class TranscriptionError(RuntimeError):
    """Raised for unrecoverable errors inside the transcription pipeline."""


# ─────────────────────────  Main helper class  ─────────────────────────
class WhisperXConfigurableTranscription:
    """
    Thread- and coroutine-safe helper around WhisperX.

    Create **one instance per worker process** (or per request, if you prefer
    total isolation).  Heavy CPU/GPU work is shunted to default executors
    so your asyncio event-loop keeps breathing.
    """

    def __init__(self, use_gpu: bool = True):
        # Set device based on CUDA availability and use_gpu parameter
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Running with CUDA on device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            if use_gpu:
                logger.warning("CUDA requested but not available - falling back to CPU")
            else:
                logger.info("Running in CPU-only mode")

        # mutable shared state guarded by asyncio locks
        self.model = None
        self._model_lock = asyncio.Lock()
        self.align_models: Dict[str, Tuple[Any, Any]] = {}
        self._align_lock: Dict[str, asyncio.Lock] = {}

    # ─────────────────────────  Private helpers  ────────────────────────
    def _fix_dtype_for_cpu(self, config: TranscriptionConfig) -> None:
        if self.device == "cpu" and config.compute_type != "float32":
            logger.warning(
                "compute_type %s not supported on CPU – switching to float32",
                config.compute_type,
            )
            config.compute_type = "float32"
        elif self.device == "cuda" and config.compute_type == "float32":
            logger.info("Switching to float16 for better GPU performance")
            config.compute_type = "float16"

    def _load_model(self, config: TranscriptionConfig):
        """Blocking model loader – run in executor."""
        self._fix_dtype_for_cpu(config)

        asr_opts = dict(
            temperatures=[config.temperature],
            beam_size=config.beam_size,
            word_timestamps=config.word_timestamps,
            condition_on_previous_text=config.condition_on_previous_text,
            compression_ratio_threshold=config.compression_ratio_threshold,
            no_speech_threshold=config.no_speech_threshold,
        )

        vad_opts = (
            dict(vad_onset=config.vad_onset, vad_offset=config.vad_offset)
            if config.vad_filter
            else None
        )

        logger.info(
            "Loading WhisperX model '%s' on %s (%s)",
            config.model_name,
            self.device,
            config.compute_type,
        )
        return whisperx.load_model(
            config.model_name,
            device=self.device,
            compute_type=config.compute_type,
            language=config.language,
            asr_options=asr_opts,
            vad_options=vad_opts,
        )

    async def _ensure_model_loaded(self, config: TranscriptionConfig):
        """Load the ASR model once per process, with locking."""
        if self.model is not None:
            return

        async with self._model_lock:
            if self.model is None:  # double-check after acquiring
                loop = asyncio.get_running_loop()
                self.model = await loop.run_in_executor(
                    None, self._load_model, config
                )
                logger.info("WhisperX model ready.")

    async def _get_align_model(self, lang: str):
        """Return (model, metadata) for a language, loading it if needed."""
        if lang in self.align_models:
            return self.align_models[lang]

        # obtain (or create) a per-language lock
        lock = self._align_lock.setdefault(lang, asyncio.Lock())

        async with lock:
            if lang in self.align_models:
                return self.align_models[lang]

            loop = asyncio.get_running_loop()

            def _load():
                logger.info("Downloading alignment model for '%s'…", lang)
                return whisperx.load_align_model(lang, device=self.device)

            model_a, meta = await loop.run_in_executor(None, _load)
            self.align_models[lang] = (model_a, meta)
            logger.info("Alignment model '%s' cached.", lang)
            return model_a, meta

    # ────────────────────────  Public API methods  ──────────────────────
    async def transcribe_audio(
        self, audio_path: str, speaker_tag: str, config: TranscriptionConfig
    ) -> Dict[str, Any]:
        """
        Transcribe one audio file; `speaker_tag` is either 'local' or 'remote'.
        """
        if not os.path.exists(audio_path):
            raise TranscriptionError(f"File not found: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise TranscriptionError(f"File is empty: {audio_path}")

        await self._ensure_model_loaded(config)

        # load wav/ogg/flac - returns np.ndarray
        audio = whisperx.load_audio(audio_path)
        logger.debug("Loaded %.2f MB audio from %s", os.path.getsize(audio_path) / 2**20, audio_path)

        # heavy inference – run in executor
        loop = asyncio.get_running_loop()
        logger.info("Running WhisperX transcription …")
        result = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio,
                batch_size=config.batch_size,
                language=config.language,
                task="transcribe",
            ),
        )
        logger.info("Transcription done.")

        detected_lang = result.get("language", config.language or "de")
        logger.debug("Detected language: %s", detected_lang)

        # free ASR GPU memory before alignment (helps on 8 GB GPUs)
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        # alignment (optional)
        segments = result["segments"]
        if config.align_output:
            try:
                model_a, meta = await self._get_align_model(detected_lang)
            except Exception as exc:  # fallback to German
                logger.warning("Align-model failed: %s – falling back to 'de'", exc)
                model_a, meta = await self._get_align_model("de")

            logger.info("Running alignment …")
            segments = await loop.run_in_executor(
                None,
                lambda: whisperx.align(
                    segments,
                    model_a,
                    meta,
                    audio,
                    device=self.device,
                    return_char_alignments=False,
                )["segments"],
            )

        # speaker labelling & full text aggregation
        label = "agent" if speaker_tag == "local" else "client"
        full_text = []
        for s in segments:
            s["speaker"] = label
            full_text.append(s.get("text", ""))

        return {"segments": segments, "text": " ".join(full_text).strip()}

    async def transcribe_audio_combined(
        self,
        local_audio_path: str,
        remote_audio_path: str,
        config: TranscriptionConfig,
    ) -> Dict[str, Any]:
        """
        Transcribe two mono recordings (local+remote) and merge chronologically.
        """
        # parallel launch – they'll share the cached model
        local_task = asyncio.create_task(
            self.transcribe_audio(local_audio_path, "local", config)
        )
        remote_task = asyncio.create_task(
            self.transcribe_audio(remote_audio_path, "remote", config)
        )
        transcript_local, transcript_remote = await asyncio.gather(local_task, remote_task)

        # combine & sort by start time
        all_segments = [
            *transcript_remote["segments"],
            *transcript_local["segments"],
        ]
        all_segments.sort(key=lambda seg: seg["start"])

        combined_text = " ".join(seg["text"] for seg in all_segments)

        return {
            "transcript_remote": transcript_remote,
            "transcript_local": transcript_local,
            "transcript_combined": {
                "segments": all_segments,
                "text": combined_text,
            },
        }
