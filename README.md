# SoulX-FlashHead RunPod Serverless

전세계 talking-head 추론용 RunPod Serverless 패키지.

## 구조
- `Dockerfile` — runpod/pytorch 2.7.1 + cu128 + ffmpeg + SoulX 의존성
- `handler.py` — RunPod handler. base64 portrait+audio → base64 mp4
- `source/` — SoulX-FlashHead 코드 (모델 weights 제외)
- 모델은 RunPod **Network Volume** 에 lazy download (`/runpod-volume/models/`)

## 빌드 + 푸시
```bash
# Docker Hub 사용 시 (예: profewoov/soulx-flashhead)
docker buildx build --platform linux/amd64 -t profewoov/soulx-flashhead:latest --push .
```

## RunPod Endpoint 설정
1. **Network Volume** 생성 — 30GB 이상 (모델 16GB + 여유), region 선택 (KR-GW 권장 시 한국 latency 짧음)
2. **Serverless Endpoint** 생성:
   - Image: `profewoov/soulx-flashhead:latest`
   - GPU: RTX 4090 24GB (또는 A6000 48GB) — 1.3B 모델은 4090 충분
   - Idle timeout: 30s
   - Max workers: 2~3
   - Network Volume mount: 위 생성한 볼륨 → `/runpod-volume`
3. 첫 호출 시 모델 16GB 자동 다운로드 (~3~10분, 한 번만)

## 호출 예시
```bash
curl -X POST https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "portrait_base64": "...",
      "audio_base64": "...",
      "model_type": "lite",
      "seed": 9999,
      "use_face_crop": false,
      "lora_choice": "OFF (base Lite)"
    }
  }'
```

응답:
```json
{
  "output": {
    "video_base64": "...",
    "duration_sec": 12.3,
    "segments": 5,
    "size_bytes": 1234567
  }
}
```

## 챗봇 cutover
`/Volumes/무제/두리컴/챗봇/.env`:
```
RUNPOD_ENDPOINT_ID=xxxxxxxxxx
SOULX_PROVIDER=runpod   # default: moon1 (fallback)
```
`talking-video-route.ts` 가 `RUNPOD_ENDPOINT_ID` 있으면 RunPod, 없으면 moon1로 분기 (TODO).
