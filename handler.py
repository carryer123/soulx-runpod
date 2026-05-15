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
import json
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


def _elapsed_ms(start):
    return int(round((time.perf_counter() - start) * 1000))


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


def _ensure_pipeline_loaded(model_type="lite", timings=None):
    """SoulX pipeline import + model load. Idempotent."""
    global _PIPELINE_LOADED
    timings = timings if timings is not None else {}

    t = time.perf_counter()
    _ensure_models()
    timings["ensure_models_ms"] = _elapsed_ms(t)

    t = time.perf_counter()
    import gradio_app_streaming as soulx_streaming
    timings["import_streaming_ms"] = _elapsed_ms(t)

    if (
        _PIPELINE_LOADED
        and soulx_streaming.pipeline is not None
        and soulx_streaming.loaded_ckpt_dir == CKPT_DIR
        and soulx_streaming.loaded_wav2vec_dir == WAV2VEC_DIR
        and soulx_streaming.loaded_model_type == model_type
    ):
        timings["pipeline_load_ms"] = 0
        return

    t = time.perf_counter()
    soulx_streaming.pipeline = soulx_streaming.get_pipeline(
        world_size=1,
        ckpt_dir=CKPT_DIR,
        model_type=model_type,
        wav2vec_dir=WAV2VEC_DIR,
    )
    soulx_streaming.loaded_ckpt_dir = CKPT_DIR
    soulx_streaming.loaded_wav2vec_dir = WAV2VEC_DIR
    soulx_streaming.loaded_model_type = model_type
    timings["pipeline_load_ms"] = _elapsed_ms(t)
    _PIPELINE_LOADED = True


def _upload_output_file(path, upload_url, headers=None):
    import urllib.request

    data = Path(path).read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="PUT",
        headers={
            "Content-Type": "video/mp4",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = getattr(resp, "status", 200)
        if status >= 400:
            raise RuntimeError(f"output upload failed: HTTP {status}")
    return len(data)


def handler(event):
    t0 = time.perf_counter()
    inp = event.get("input") or {}
    timings = {}

    portrait_b64 = inp.get("portrait_base64")
    audio_b64 = inp.get("audio_base64")
    if not portrait_b64 or not audio_b64:
        return {"error": "portrait_base64 and audio_base64 required"}

    model_type = inp.get("model_type", "lite")
    seed = int(inp.get("seed", 9999))
    use_face_crop = bool(inp.get("use_face_crop", False))
    lora_choice = inp.get("lora_choice", "OFF (base Lite)")
    output_upload_url = inp.get("output_upload_url")
    output_url = inp.get("output_url")
    output_upload_headers = inp.get("output_upload_headers") or {}
    return_video_base64 = bool(inp.get("return_video_base64", not output_upload_url))

    tmp = tempfile.mkdtemp(prefix="soulx_")
    portrait_path = os.path.join(tmp, "portrait.jpg")
    audio_path = os.path.join(tmp, "audio.wav")
    t = time.perf_counter()
    portrait_bytes = base64.b64decode(portrait_b64)
    audio_bytes = base64.b64decode(audio_b64)
    timings["base64_decode_ms"] = _elapsed_ms(t)
    timings["portrait_bytes"] = len(portrait_bytes)
    timings["audio_bytes"] = len(audio_bytes)

    t = time.perf_counter()
    Path(portrait_path).write_bytes(portrait_bytes)
    Path(audio_path).write_bytes(audio_bytes)
    timings["input_write_ms"] = _elapsed_ms(t)

    try:
        _ensure_pipeline_loaded(model_type=model_type, timings=timings)
        import gradio_app_streaming as soulx_streaming

        segments = []
        inference_metrics = {}
        t = time.perf_counter()
        for seg in soulx_streaming.run_inference_streaming(
            ckpt_dir=CKPT_DIR,
            wav2vec_dir=WAV2VEC_DIR,
            model_type=model_type,
            cond_image=portrait_path,
            audio_path=audio_path,
            seed=seed,
            use_face_crop=use_face_crop,
            lora_choice=lora_choice,
            metrics=inference_metrics,
            mux_segment_audio=False,
            save_final=False,
        ):
            segments.append(seg)
        timings["streaming_generator_ms"] = _elapsed_ms(t)

        if not segments:
            return {"error": "no segments produced"}

        out_id = uuid.uuid4().hex[:12]
        out_path = os.path.join(OUT_DIR, f"talking_{out_id}.mp4")
        list_path = os.path.join(tmp, "concat.txt")
        with open(list_path, "w") as f:
            for s in segments:
                f.write(f"file '{s}'\n")

        # video-only concat + remux original wav (avoid segment-boundary audio glitch)
        t = time.perf_counter()
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
        timings["final_concat_ms"] = _elapsed_ms(t)

        size_bytes = os.path.getsize(out_path)
        result = {
            "duration_sec": round(time.perf_counter() - t0, 2),
            "segments": len(segments),
            "size_bytes": size_bytes,
            "timings_ms": timings,
            "inference_metrics": inference_metrics,
        }

        if output_upload_url:
            t = time.perf_counter()
            uploaded_bytes = _upload_output_file(out_path, output_upload_url, output_upload_headers)
            timings["output_upload_ms"] = _elapsed_ms(t)
            result["uploaded_bytes"] = uploaded_bytes
            if output_url:
                result["video_url"] = output_url

        if return_video_base64:
            t = time.perf_counter()
            video_bytes = Path(out_path).read_bytes()
            timings["output_read_ms"] = _elapsed_ms(t)
            t = time.perf_counter()
            result["video_base64"] = base64.b64encode(video_bytes).decode("ascii")
            timings["base64_encode_ms"] = _elapsed_ms(t)

        try:
            os.remove(out_path)
        except OSError:
            pass

        print("[soulx] job_metrics " + json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"inference failed: {e}"}
    finally:
        try:
            shutil.rmtree(tmp)
        except OSError:
            pass


if os.environ.get("SOULX_PRELOAD_MODEL", "0").lower() not in {"0", "false", "no"}:
    _preload_timings = {}
    _preload_t = time.perf_counter()
    try:
        _ensure_pipeline_loaded(
            model_type=os.environ.get("SOULX_PRELOAD_MODEL_TYPE", "lite"),
            timings=_preload_timings,
        )
        print(
            "[soulx] preload_metrics "
            + json.dumps(
                {
                    "total_ms": _elapsed_ms(_preload_t),
                    **_preload_timings,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception:
        import traceback

        print("[soulx] preload_failed", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
