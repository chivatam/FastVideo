"""VLM-based scene matching using AVoCaDO (Qwen2.5-Omni).

Replaces VBench's Tag2Text-based scene metric with a modern VLM caption.
The algorithm follows VBench:
  1. Caption the video
  2. Check if all scene keywords appear in the caption
  3. Score = 1.0 if all match, 0.0 otherwise

Unlike VBench (which captions each frame separately with Tag2Text),
AVoCaDO captions the entire video in one pass with rich natural language,
making the keyword check more robust.
"""

from __future__ import annotations

from typing import Any

import os
import tempfile

import av
import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_SCENE_PROMPT = ("Describe the visual scene in this video, including the location, "
                 "environment, objects, and overall setting. Be specific and use "
                 "concrete descriptive words.")


def _install_torchvision_read_video_compat() -> None:
    """Restore torchvision.io.read_video using PyAV on torchvision >= 0.24.

    qwen-omni-utils still calls the legacy torchvision API as its fallback
    video reader. Recent torchvision releases removed that API in favor of
    TorchCodec, whose binary wheels are not yet compatible with this host's
    PyTorch/CUDA stack. This shim preserves the legacy return contract and
    leaves qwen-omni-utils' frame selection unchanged.
    """
    import torchvision.io

    if hasattr(torchvision.io, "read_video"):
        return

    def _read_video(
        filename: str,
        start_pts: float = 0.0,
        end_pts: float | None = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if pts_unit not in {"pts", "sec"}:
            raise ValueError(f"unsupported pts_unit={pts_unit!r}")
        if filename.startswith("file://"):
            filename = filename[7:]

        frames: list[np.ndarray] = []
        with av.open(filename) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            video_fps = float(rate) if rate is not None else 0.0
            time_base = float(stream.time_base)

            for frame in container.decode(stream):
                frame_pts = float(frame.pts or 0)
                frame_time = float(frame.time) if frame.time is not None else frame_pts * time_base
                position = frame_time if pts_unit == "sec" else frame_pts
                if position < float(start_pts):
                    continue
                if end_pts is not None and position > float(end_pts):
                    break
                frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            raise RuntimeError(f"no video frames decoded from {filename}")

        video = torch.from_numpy(np.stack(frames, axis=0))
        if output_format == "TCHW":
            video = video.permute(0, 3, 1, 2).contiguous()
        elif output_format != "THWC":
            raise ValueError(f"unsupported output_format={output_format!r}")

        audio = torch.empty((1, 0), dtype=torch.float32)
        return video, audio, {"video_fps": video_fps, "audio_fps": 0.0}

    torchvision.io.read_video = _read_video


@register("vbench.scene")
class SceneMetric(BaseMetric):

    name = "vbench.scene"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["transformers", "qwen_omni_utils"]
    backbone = "avocado"

    def __init__(self, model_path: str = "AVoCaDO-Captioner/AVoCaDO") -> None:
        super().__init__()
        self._model: Any = None
        self._processor: Any = None
        self._model_path = model_path

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        # AVoCaDO uses Qwen2.5-Omni — large multimodal model
        os.environ.setdefault("VIDEO_MAX_PIXELS", str(20070400))  # 512*28*28*50

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map=str(self.device) if self.device.type == "cuda" else None,
        )
        self._model.disable_talker()
        self._model.eval()
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self._model_path)

    def _save_temp_video(self, video: torch.Tensor) -> str:
        """Save (T, C, H, W) float [0,1] tensor as a temp mp4 file."""
        frames = (
            (video * 255)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        # Caller owns the resulting file (it's read by Qwen2.5-Omni and
        # cleaned up at end of compute()), so we just need a unique path.
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with av.open(path, mode="w") as container:
            stream = container.add_stream("libx264", rate=8)
            stream.width = int(frames.shape[2])
            stream.height = int(frames.shape[1])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "ultrafast"}
            for frame in frames:
                video_frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(frame),
                    format="rgb24",
                )
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    def _generate_caption(self, video_path: str) -> str:
        # Force qwen-omni-utils onto the compatibility reader instead of
        # probing TorchCodec, which cannot load against this PyTorch build.
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
        _install_torchvision_read_video_compat()

        from qwen_omni_utils import process_mm_info
        from qwen_omni_utils.v2_5 import vision_process

        vision_process.FORCE_QWENVL_VIDEO_READER = "torchvision"
        vision_process.get_video_reader_backend.cache_clear()

        conversation = [
            {
                "role":
                "system",
                "content": [{
                    "type":
                    "text",
                    "text": ("You are Qwen, a virtual human developed by the Qwen Team, "
                             "Alibaba Group, capable of perceiving auditory and visual inputs.")
                }],
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 401408
                    },
                    {
                        "type": "text",
                        "text": _SCENE_PROMPT
                    },
                ],
            },
        ]

        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # AVoCaDO is video+audio; for scene matching we don't need audio,
        # but the model expects it so let it process
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        with torch.no_grad():
            # Transformers 5.2 still dereferences ``self.talker`` inside the
            # wrapper even for text-only generation after disable_talker().
            # Calling the wrapper's first stage directly is equivalent to its
            # text-only path and avoids retaining the unused speech decoder.
            text_ids = self._model.thinker.generate(
                **inputs,
                use_audio_in_video=False,
                do_sample=False,
                max_new_tokens=512,
            )

        decoded = self._processor.batch_decode(text_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
        return decoded.split("\nassistant\n")[-1].lower()

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        video = sample["video"]  # (T, C, H, W)
        aux = sample.get("auxiliary_info") or {}
        if "scene" not in aux:
            return self._skip(sample, "missing 'scene' in auxiliary_info")

        scene_keywords = aux["scene"]
        # VBench's full-info manifest stores this dimension as
        # {"scene": {"scene": "alley"}}, while older callers supplied the
        # inner string directly.
        if isinstance(scene_keywords, dict):
            scene_keywords = scene_keywords.get("scene", "")
        if not isinstance(scene_keywords, str) or not scene_keywords.strip():
            return self._skip(sample, "invalid 'scene' in auxiliary_info")
        keywords = [k.strip().lower() for k in scene_keywords.split() if k.strip()]

        tmp_path = self._save_temp_video(video)
        try:
            caption = self._generate_caption(tmp_path)
        finally:
            os.unlink(tmp_path)

        matched = [kw for kw in keywords if kw in caption]
        score = 1.0 if len(matched) == len(keywords) else 0.0
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "caption": caption[:500],
                "keywords": keywords,
                "matched": matched,
            },
        )
