"""
RunPod Serverless handler for SoulX-FlashHead Lite.
Input (event["input"]):
  - portrait_base64: str (jpg/png)
  - audio_base64: str (wav, 16k mono recommended)
  - model_type: "lite" (default) | "pro"
  - seed: int (default 9999)
  - use_face_crop: bool (default False)
  - lora_choice: str (default "OFF (base Lite)")
Output:
  - video_base64: str (mp4)
  - duration_sec: float
  - segments: int
"""
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import runpod

MODELS_DIR = os.environ.get("SOULX_MODELS_DIR", "/runpod-volume/models")
CKPT_DIR = os.path.join(MODELS_DIR, "SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.path.join(MODELS_DIR, "wav2vec2-base-960h")
OUT_DIR = "/tmp/soulx_out"
os.makedirs(OUT_DIR, exist_ok=True)

_PIPELINE_LOADED = False


def _ensure_models():
    """Lazy download models to Network Volume on first cold start."""
    from huggingface_hub import snapshot_download

    if not os.path.isdir(CKPT_DIR) or not os.listdir(CKPT_DIR):
        print(f"[soulx] downloading SoulX-FlashHead-1_3B → {CKPT_DIR}", flush=True)
        snapshot_download(
            repo_id="Soul-AILab/SoulX-FlashHead-1_3B",
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
        )
    if not os.path.isdir(WAV2VEC_DIR) or not os.listdir(WAV2VEC_DIR):
        print(f"[soulx] downloading wav2vec2-base-960h → {WAV2VEC_DIR}", flush=True)
        snapshot_download(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=WAV2VEC_DIR,
            local_dir_use_symlinks=False,
        )


def _ensure_pipeline_loaded():
    """SoulX pipeline import + warmup. Idempotent."""
    global _PIPELINE_LOADED
    if _PIPELINE_LOADED:
        return
    _ensure_models()
    # gradio_app_streaming 안의 run_inference_streaming은 첫 호출 시 pipeline init
    import gradio_app_streaming  # noqa: F401
    _PIPELINE_LOADED = True


def handler(event):
    t0 = time.time()
    inp = event.get("input") or {}

    portrait_b64 = inp.get("portrait_base64")
    audio_b64 = inp.get("audio_base64")
    if not portrait_b64 or not audio_b64:
        return {"error": "portrait_base64 and audio_base64 required"}

    model_type = inp.get("model_type", "lite")
    seed = int(inp.get("seed", 9999))
    use_face_crop = bool(inp.get("use_face_crop", False))
    lora_choice = inp.get("lora_choice", "OFF (base Lite)")

    tmp = tempfile.mkdtemp(prefix="soulx_")
    portrait_path = os.path.join(tmp, "portrait.jpg")
    audio_path = os.path.join(tmp, "audio.wav")
    Path(portrait_path).write_bytes(base64.b64decode(portrait_b64))
    Path(audio_path).write_bytes(base64.b64decode(audio_b64))

    try:
        _ensure_pipeline_loaded()
        from gradio_app_streaming import run_inference_streaming

        segments = []
        for seg in run_inference_streaming(
            ckpt_dir=CKPT_DIR,
            wav2vec_dir=WAV2VEC_DIR,
            model_type=model_type,
            cond_image=portrait_path,
            audio_path=audio_path,
            seed=seed,
            use_face_crop=use_face_crop,
            lora_choice=lora_choice,
        ):
            segments.append(seg)

        if not segments:
            return {"error": "no segments produced"}

        out_id = uuid.uuid4().hex[:12]
        out_path = os.path.join(OUT_DIR, f"talking_{out_id}.mp4")
        list_path = os.path.join(tmp, "concat.txt")
        with open(list_path, "w") as f:
            for s in segments:
                f.write(f"file '{s}'\n")

        # video-only concat + remux original wav (avoid segment-boundary audio glitch)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart", out_path,
            ],
            check=True,
        )

        video_bytes = Path(out_path).read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode("ascii")

        try:
            os.remove(out_path)
        except OSError:
            pass

        return {
            "video_base64": video_b64,
            "duration_sec": round(time.time() - t0, 2),
            "segments": len(segments),
            "size_bytes": len(video_bytes),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"inference failed: {e}"}
    finally:
        try:
            shutil.rmtree(tmp)
        except OSError:
            pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
