"""
Gradio 流式视频生成：视频生成&视频保存异步进行，确保实时性
"""
import gradio as gr
import os
import torch
import numpy as np
import time
import wave
import imageio
import librosa
import subprocess
import queue
import threading
from datetime import datetime
from collections import deque
from loguru import logger

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

# gr.Video 的 streaming=True 要求视频片段大于1s，实际需要接近3s才能不卡顿。
# 为了适配，每 3 个 chunk 合并为一段视频
CHUNKS_PER_SEGMENT = 3

pipeline = None
loaded_ckpt_dir = None
loaded_wav2vec_dir = None
loaded_model_type = None


def _write_frames_to_mp4(frames_list, video_path, fps):
    """将帧列表写入 MP4（仅视频轨）。"""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    with imageio.get_writer(
        video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                writer.append_data(frames_np[i, :, :, :])
    return video_path


def save_video_with_audio(frames_list, video_path, audio_path, fps):
    """写入完整视频并混入完整音频（-shortest 保证音画同步，yuv420p + faststart 保证浏览器可播）。"""
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    _write_frames_to_mp4(frames_list, temp_path, fps)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            # "-shortest",
            video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return video_path

def _save_chunk_audio_to_wav(audio_array, wav_path, sample_rate=16000):
    """将一段 float32 [-1,1] 的音频数组保存为 wav 文件。"""
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return wav_path

def run_inference_streaming(
    ckpt_dir,
    wav2vec_dir,
    model_type,
    cond_image,
    audio_path,
    seed,
    use_face_crop,
    lora_choice,
    progress=gr.Progress(),
):
    """
    流式推理：主程序监控 res_queue，有 frames 就保存并 yield；
    推理在独立线程中执行，按 chunk 顺序 infer，结果放入 res_queue。
    """
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type

    if (
        pipeline is None
        or loaded_ckpt_dir != ckpt_dir
        or loaded_wav2vec_dir != wav2vec_dir
        or loaded_model_type != model_type
    ):
        progress(0.2, desc="Loading Model...")
        logger.info(f"Loading pipeline with ckpt_dir={ckpt_dir}, wav2vec_dir={wav2vec_dir}")
        try:
            pipeline = get_pipeline(
                world_size=1,
                ckpt_dir=ckpt_dir,
                model_type=model_type,
                wav2vec_dir=wav2vec_dir,
            )
            loaded_ckpt_dir = ckpt_dir
            loaded_wav2vec_dir = wav2vec_dir
            loaded_model_type = model_type
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise gr.Error(f"Failed to load model: {e}")

    apply_lora_state(pipeline, LORA_CHOICES.get(lora_choice))
    progress(0.5, desc="Preparing Data...")
    base_seed = int(seed) if seed >= 0 else 9999
    try:
        get_base_data(
            pipeline,
            cond_image_path_or_dir=cond_image,
            base_seed=base_seed,
            use_face_crop=use_face_crop,
        )
    except Exception as e:
        logger.error(f"Error in get_base_data: {e}")
        raise gr.Error(f"Error processing inputs: {e}")

    infer_params = get_infer_params()
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    try:
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise gr.Error(f"Failed to load audio file: {e}")

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

    stream_dir = os.path.join("gradio_results", "stream_preview")
    os.makedirs(stream_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    accumulated = []

    # 默认使用 stream 模式：准备 chunk 切片
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    remainder = len(human_speech_array_all) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        human_speech_array_all = np.concatenate(
            [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
        )
    human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)
    total_chunks = len(human_speech_array_slices)
    if total_chunks == 0:
        raise gr.Error("Audio too short: no chunks to generate. Please use a longer audio.")

    # Data prepare：按每 k 个 chunk 合并为一段 wav 保存（时间戳+segment_id 命名）
    segment_audio_paths = {}
    num_segments = (total_chunks + CHUNKS_PER_SEGMENT - 1) // CHUNKS_PER_SEGMENT
    for segment_id in range(num_segments):
        start = segment_id * CHUNKS_PER_SEGMENT
        end = min(start + CHUNKS_PER_SEGMENT, total_chunks)
        audio_concat = np.concatenate(
            [human_speech_array_slices[i] for i in range(start, end)]
        )
        segment_audio_name = f"audio_{timestamp}_seg_{segment_id:04d}.wav"
        segment_audio_path = os.path.join(stream_dir, segment_audio_name)
        _save_chunk_audio_to_wav(
            audio_concat,
            segment_audio_path,
            sample_rate=sample_rate,
        )
        segment_audio_paths[segment_id] = segment_audio_path
    logger.info(
        f"Pre-saved {num_segments} segment audios (every {CHUNKS_PER_SEGMENT} chunks) under {stream_dir}"
    )

    # 结果队列：推理线程放入 (chunk_idx, chunk_frames_np)，主线程根据 chunk_id 取对应音频合并
    res_queue = queue.Queue()

    def inference_worker():
        """单独线程：按 chunk 顺序执行 infer，每生成一帧就放入 res_queue，立即继续下一 chunk。"""
        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
            torch.cuda.synchronize()
            start_time = time.time()
            video = run_pipeline(pipeline, audio_embedding)
            video = video[motion_frames_num:]
            torch.cuda.synchronize()
            logger.info(f"Infer chunk-{chunk_idx} done, cost time: {time.time() - start_time:.2f}s")
            chunk_frames_np = video.cpu().numpy()
            res_queue.put((chunk_idx, chunk_frames_np))
        res_queue.put(None)  # 结束哨兵

    worker_thread = threading.Thread(target=inference_worker)
    worker_thread.start()
    logger.info("Inference worker thread started. Main will consume res_queue and yield video paths.")

    # 主程序：监控 res_queue，每凑满 k 个 chunk 合并为一段 mp4（含对应段音频）并 yield
    frame_buffer = []
    while True:
        item = res_queue.get()
        if item is None:
            break
        chunk_idx, chunk_frames_np = item
        chunk_frames = torch.from_numpy(chunk_frames_np)
        accumulated.append(chunk_frames)
        frame_buffer.append(chunk_frames)
        if len(frame_buffer) == CHUNKS_PER_SEGMENT:
            segment_id = (chunk_idx + 1 - CHUNKS_PER_SEGMENT) // CHUNKS_PER_SEGMENT
            segment_audio_path = segment_audio_paths[segment_id]
            segment_path = os.path.join(
                stream_dir, f"preview_{timestamp}_seg_{segment_id:04d}.mp4"
            )
            save_video_with_audio(
                frame_buffer,
                segment_path,
                segment_audio_path,
                fps=tgt_fps,
            )
            logger.info(
                f"Saved segment-{segment_id} (chunks {segment_id * CHUNKS_PER_SEGMENT}-{chunk_idx}) and yielding to frontend."
            )
            yield os.path.abspath(segment_path)
            frame_buffer = []

    # 不足 k 的剩余 chunk 合并为最后一段
    if frame_buffer:
        segment_id = num_segments - 1
        segment_audio_path = segment_audio_paths[segment_id]
        segment_path = os.path.join(
            stream_dir, f"preview_{timestamp}_seg_{segment_id:04d}.mp4"
        )
        save_video_with_audio(
            frame_buffer,
            segment_path,
            segment_audio_path,
            fps=tgt_fps,
        )
        logger.info(
            f"Saved final segment-{segment_id} ({len(frame_buffer)} chunks) and yielding to frontend."
        )
        yield os.path.abspath(segment_path)

    worker_thread.join()

    if not accumulated:
        raise gr.Error("No video frames generated. Please check inputs and try again.")

    output_dir = "gradio_results"
    os.makedirs(output_dir, exist_ok=True)
    final_filename = f"res_{timestamp}.mp4"
    final_path = os.path.join(output_dir, final_filename)
    save_video_with_audio(accumulated, final_path, audio_path, fps=tgt_fps)
    logger.info(f"Saved to {final_path}")




# ===== Voice TTS (ElevenLabs) =====
import urllib.request
import re
import json as _json
import tempfile

ELEVENLABS_API_KEY = "sk_c6f79eb3747a7e8139ee3ed97ae8ad6ebb083ef9016a6eec"
VOICES = {
    "노아 (ElevenLabs)": ("eleven", "rnQbo1wPfI0D7fhlH3ca"),
    "도희 (ElevenLabs)": ("eleven", "uVF4UWp8zsIkCNPxG4Ca"),
    "이수 (ElevenLabs)": ("eleven", "yoxuHpVE5CXlx3EBROzv"),
    "Kore (Gemini · 따뜻 여성)": ("gemini", "Kore"),
    "Aoede (Gemini · 산뜻 여성)": ("gemini", "Aoede"),
    "Leda (Gemini · 젊은 여성)": ("gemini", "Leda"),
    "Zephyr (Gemini · 밝은 여성)": ("gemini", "Zephyr"),
    "Puck (Gemini · 활발 남성)": ("gemini", "Puck"),
    "Charon (Gemini · 차분 남성)": ("gemini", "Charon"),
    "Orus (Gemini · 단단 남성)": ("gemini", "Orus"),
    "Fenrir (Gemini · 에너지 남성)": ("gemini", "Fenrir"),
}

ELEVENLABS_API_KEY = "sk_c6f79eb3747a7e8139ee3ed97ae8ad6ebb083ef9016a6eec"
GEMINI_API_KEY = "AIzaSyDjcVydb5qEq2yvZ1WokBW-i7l3NIT7Xro"
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"


def _add_pauses_ssml(text):
    """Add ElevenLabs SSML break tags after punctuation for natural pacing."""
    text = re.sub(r"([.!?])\s*", r"\1 <break time=\"0.45s\"/> ", text)
    text = re.sub(r"([,])\s*", r"\1 <break time=\"0.25s\"/> ", text)
    return text


def _tts_eleven(text, voice_id, model_id="eleven_multilingual_v2"):
    paced = _add_pauses_ssml(text)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    body = _json.dumps({
        "text": paced,
        "model_id": model_id,
        "voice_settings": {"stability": 0.6, "similarity_boost": 0.85, "style": 0.15, "use_speaker_boost": True}
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return ("mp3", resp.read())


def _tts_gemini(text, voice_name):
    import base64
    paced = re.sub(r"([.!?])\s*", r"\1\n", text)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = _json.dumps({
        "contents": [{"parts": [{"text": paced}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}}
        }
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = _json.loads(resp.read())
    parts = d["candidates"][0]["content"]["parts"]
    for p_ in parts:
        idata = p_.get("inline_data") or p_.get("inlineData")
        if idata:
            return ("pcm24k", base64.b64decode(idata["data"]))
    raise RuntimeError("no audio in Gemini response")


def tts_generate(text, voice_label):
    if not text or not text.strip():
        return None
    backend, vid = VOICES.get(voice_label, list(VOICES.values())[0])
    if backend == "eleven":
        kind, raw = _tts_eleven(text, vid, "eleven_v3")
    else:
        kind, raw = _tts_gemini(text, vid)

    src_path = tempfile.mktemp(suffix=("." + ("mp3" if kind == "mp3" else "raw")), prefix="tts_")
    with open(src_path, "wb") as f:
        f.write(raw)
    wav_path = src_path.rsplit(".", 1)[0] + "_16k.wav"
    import subprocess as _sp
    if kind == "mp3":
        _sp.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path], check=True)
    else:
        _sp.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path], check=True)
    return wav_path



# ===== LoRA toggle =====
LORA_CHOICES = {
    "OFF (base Lite)": None,
    "v1 r16 1k": "/home/moon1/SoulX-FlashHead/lora_out_v1/final",
    "v2 r64 1k": "/home/moon1/SoulX-FlashHead/lora_out_v2_r64/final",
    "v4 r64 2k lr3e-5": "/home/moon1/SoulX-FlashHead/lora_out_v4_r64_lr3e5/step_002000",
    "v5 r8 2k cosine": "/home/moon1/SoulX-FlashHead/lora_out_v5_r8_5k_cos/step_002000",
    "v6 small r16 1k": "/home/moon1/SoulX-FlashHead/lora_out_v6_r16_5k_small/step_001000",
    "v6 small r16 2k": "/home/moon1/SoulX-FlashHead/lora_out_v6_r16_5k_small/step_002000",
    "v5 r8 5k final": "/home/moon1/SoulX-FlashHead/lora_out_v5_r8_5k_cos/final",
}
_lora_state = {"loaded_path": None, "wrapped": False}

def apply_lora_state(pipe, lora_path):
    """Ensure pipeline.model has the requested LoRA active (or none)."""
    if lora_path is None:
        if _lora_state["wrapped"]:
            try:
                pipe.model.disable_adapter_layers()
                logger.info("[LoRA] disabled")
            except Exception as e:
                logger.warning(f"[LoRA] disable failed: {e}")
        return
    # need to load this specific LoRA
    if _lora_state["loaded_path"] == lora_path and _lora_state["wrapped"]:
        try:
            pipe.model.enable_adapter_layers()
            logger.info(f"[LoRA] re-enabled {lora_path}")
        except Exception as e:
            logger.warning(f"[LoRA] enable failed: {e}")
        return
    # different LoRA or not yet wrapped: wrap fresh
    from peft import PeftModel
    if _lora_state["wrapped"]:
        # unwrap current adapter to get back base model
        try:
            base = pipe.model.get_base_model()
        except Exception:
            base = pipe.model.base_model.model if hasattr(pipe.model, "base_model") else pipe.model
        pipe.model = base
    pipe.model = PeftModel.from_pretrained(pipe.model, lora_path)
    pipe.model.eval()
    _lora_state["loaded_path"] = lora_path
    _lora_state["wrapped"] = True
    logger.info(f"[LoRA] loaded {lora_path}")

# ---------- Gradio UI ----------
with gr.Blocks(title="혜린 라이브 아바타", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ 혜린 라이브 아바타")
    gr.Markdown("이미지와 오디오를 업로드하면 실시간 생성+재생됩니다. (단일 GPU 지원)")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎬 입력")
                with gr.Row():
                    cond_image_input = gr.Image(
                        label="혜린 사진",
                        type="filepath",
                        value="examples/haerin_portrait.jpg",
                        height=300,
                    )
                    audio_path_input = gr.Audio(
                        label="오디오 (TTS wav)",
                        type="filepath",
                        value="examples/haerin_tts_ko.wav",
                    )
            with gr.Group():
                gr.Markdown("### 🗣️ 텍스트 → 보이스 클론 → 오디오")
                voice_dropdown = gr.Dropdown(label="목소리 선택", choices=list(VOICES.keys()), value=list(VOICES.keys())[0])
                tts_text = gr.Textbox(label="대사 입력", value="안녕하세요. 반갑습니다.", lines=2)
                tts_btn = gr.Button("🎤 TTS 생성 → 오디오 자동 입력", variant="secondary")
            generate_btn = gr.Button("🚀 영상 생성 시작", variant="primary", size="lg")
            with gr.Accordion("⚙️ 고급 설정", open=False):
                ckpt_dir_input = gr.Textbox(
                    label="모델 체크포인트 경로", visible=False,
                    value="models/SoulX-FlashHead-1_3B",
                )
                wav2vec_dir_input = gr.Textbox(
                    label="Wav2Vec 경로", visible=False,
                    value="models/wav2vec2-base-960h",
                )
                model_type_input = gr.Dropdown(
                    label="모델 타입",
                    choices=["pro", "lite"],
                    value="lite",
                )
                use_face_crop_input = gr.Checkbox(label="얼굴 자동 크롭", value=False)
                lora_choice_input = gr.Dropdown(label="LoRA 적용", choices=list(LORA_CHOICES.keys()), value="OFF (base Lite)")
                seed_input = gr.Number(label="랜덤 시드", value=9999, precision=0)
        with gr.Column(scale=1):
            gr.Markdown("### 📺 출력 영상 (스트리밍)")
            video_output = gr.Video(
                label="생성된 영상",
                height=512,
                format="mp4",
                streaming=True,
                autoplay=True,
            )


    tts_btn.click(fn=tts_generate, inputs=[tts_text, voice_dropdown], outputs=audio_path_input)
    generate_btn.click(
        fn=run_inference_streaming,
        inputs=[
            ckpt_dir_input,
            wav2vec_dir_input,
            model_type_input,
            cond_image_input,
            audio_path_input,
            seed_input,
            use_face_crop_input,
            lora_choice_input,
        ],
        outputs=video_output,
    )

# (if __main__ block removed for handler import — moon1 시 외부 entry로 처리)
