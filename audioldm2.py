import os
import json
import glob
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
from diffusers import AudioLDMPipeline
from scipy.io.wavfile import write as wav_write

from audio_prompt import generate_prompts
from utils import ensure_dir, sanitize_filename


def _load_pipeline(model_id: str, hf_token: str | None) -> AudioLDMPipeline:
    """AudioLDM2 파이프라인 로드"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"AudioLDM2 모델 로딩 중... (Device: {device}, Dtype: {dtype})")
    
    pipe = AudioLDMPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_auth_token=hf_token,
    )
    pipe = pipe.to(device)
    
    print(f"✅ AudioLDM2 모델 로드 완료! Device: {pipe.device}")
    return pipe


# _ensure_dir 함수는 utils.py의 ensure_dir로 대체됨


def _load_json(path: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _objects_to_sound_sources_if_needed(data: Dict[str, Any]) -> Dict[str, Any]:
    """objects를 sound_sources로 매핑 (호환성)"""
    if "sound_sources" in data:
        return data
    if "objects" in data and isinstance(data["objects"], list):
        return {
            **data,
            "sound_sources": data["objects"],
        }
    return data


# _sanitize_filename 함수는 utils.py의 sanitize_filename으로 대체됨


def _save_wav(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """오디오를 WAV 파일로 저장"""
    # Ensure mono float32 in -1..1 -> int16
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    wav = (audio * 32767.0).astype(np.int16)
    wav_write(path, sample_rate, wav)


def generate_audio_for_sound_sources(
    sound_source_dir: str = "sound_sources",
    result_dir: str = "result",
    model_id: str = "cvssp/audioldm-s-full-v2",
    audio_seconds: float = 4.0,
    steps: int = 200,
    guidance: float = 3.5,
    seed: int | None = None,
    single: str | None = None,
) -> None:
    """Sound sources JSON 파일들을 처리하여 오디오 생성"""
    
    ensure_dir(result_dir)

    hf_token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    pipe = _load_pipeline(model_id, hf_token)

    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = torch.Generator(device=pipe.device)

    # JSON 파일들 찾기
    if single:
        if single.lower().endswith(".json"):
            candidate = single
        else:
            # 이미지 파일명에서 기본 이름 추출 (예: data/101.jpg -> 101)
            base_name = os.path.splitext(os.path.basename(single))[0]
            # sound_sources 디렉토리에서 해당 이미지의 JSON 파일 찾기
            json_pattern = os.path.join(sound_source_dir, base_name, f"{base_name}_sound_source.json")
            candidate = json_pattern if os.path.exists(json_pattern) else single
        json_files = [candidate] if os.path.exists(candidate) else []
    else:
        # sound_sources 디렉토리의 모든 이미지 폴더에서 JSON 파일 찾기
        json_files = []
        for image_folder in os.listdir(sound_source_dir):
            image_folder_path = os.path.join(sound_source_dir, image_folder)
            if os.path.isdir(image_folder_path):
                json_pattern = os.path.join(image_folder_path, "*_sound_source.json")
                json_files.extend(glob.glob(json_pattern))
        json_files = sorted(json_files)
    
    if not json_files:
        print(f"❌ JSON 파일을 찾을 수 없습니다: {sound_source_dir}")
        return

    print(f"📁 {len(json_files)}개 JSON 파일 발견. 출력 -> {result_dir}")

    total_audio_generated = 0
    
    for json_path in json_files:
        print(f"\n🎵 처리 중: {json_path}")
        
        # 이미지 이름 추출 (폴더명 또는 파일명에서)
        if os.path.dirname(json_path) != sound_source_dir:
            # 하위 폴더에 있는 경우
            base = os.path.basename(os.path.dirname(json_path))
        else:
            # 직접 sound_sources에 있는 경우
            base = os.path.splitext(os.path.basename(json_path))[0].replace("_sound_source", "")
        
        try:
            data = _objects_to_sound_sources_if_needed(_load_json(json_path))
            prompts: List[Dict[str, Any]] = generate_prompts(data)
            
            if not prompts:
                print(f"  ⚠️ {json_path}에서 프롬프트를 생성할 수 없습니다")
                continue

            # 이미지별 결과 폴더 생성
            image_out_dir = os.path.join(result_dir, base)
            ensure_dir(image_out_dir)

            print(f"  🎯 {len(prompts)}개 오디오 클립 생성 중...")
            
            for idx, item in enumerate(prompts, 1):
                prompt_text = item["prompt"]
                source_name = sanitize_filename(item.get("source_name", "source"))
                play_method = sanitize_filename(str(item.get("play_method", "act")))

                out_name = f"{idx:02d}_{source_name}_{play_method}.wav"
                out_path = os.path.join(image_out_dir, out_name)

                try:
                    with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu")):
                        audio = pipe(
                            prompt_text,
                            num_inference_steps=steps,
                            audio_length_in_s=audio_seconds,
                            guidance_scale=guidance,
                            generator=generator,
                        ).audios[0]

                    _save_wav(out_path, np.array(audio), sample_rate=16000)
                    print(f"    ✅ {out_name}")
                    total_audio_generated += 1
                    
                except Exception as e:
                    print(f"    ❌ {out_name} 생성 실패: {str(e)}")
            
            # 사용된 프롬프트들을 추적용으로 저장
            prompts_dump = os.path.join(image_out_dir, "prompts.json")
            with open(prompts_dump, "w", encoding="utf-8") as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)
            
            print(f"  📄 프롬프트 저장: {prompts_dump}")
            
        except Exception as e:
            print(f"  ❌ {json_path} 처리 중 오류: {str(e)}")
    
    print(f"\n🎉 오디오 생성 완료!")
    print(f"📊 총 {total_audio_generated}개 오디오 파일 생성")
    print(f"📁 결과 저장 위치: {result_dir}")


def run_generation(
    sound_source_dir: str = "sound_sources",
    result_dir: str = "result",
    model_id: str = "cvssp/audioldm-s-full-v2",
    audio_seconds: float = 4.0,
    steps: int = 200,
    guidance: float = 3.5,
    seed: int | None = None,
    single: str | None = None,
) -> None:
    """오디오 생성 실행"""
    try:
        generate_audio_for_sound_sources(
            sound_source_dir=sound_source_dir,
            result_dir=result_dir,
            model_id=model_id,
            audio_seconds=audio_seconds,
            steps=steps,
            guidance=guidance,
            seed=seed,
            single=single,
        )
    except Exception as e:
        print(f"❌ 오디오 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AudioLDM2를 사용한 오디오 생성")
    parser.add_argument("--src", type=str, default="sound_sources", help="Sound sources 디렉토리")
    parser.add_argument("--out", type=str, default="result", help="출력 디렉토리")
    parser.add_argument("--model", type=str, default="cvssp/audioldm-s-full-v2", help="AudioLDM 모델 ID")
    parser.add_argument("--seconds", type=float, default=4.0, help="오디오 길이 (초)")
    parser.add_argument("--steps", type=int, default=200, help="Diffusion 스텝 수")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--single", type=str, default=None, help="단일 샘플: 이미지 이름 (예: 101) 또는 JSON 파일 경로")
    
    args = parser.parse_args()

    print("🎵 AudioLDM2 오디오 생성기")
    print(f"📁 소스: {args.src}")
    print(f"📁 출력: {args.out}")
    print(f"🤖 모델: {args.model}")

    run_generation(
        sound_source_dir=args.src,
        result_dir=args.out,
        model_id=args.model,
        audio_seconds=args.seconds,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        single=args.single,
    )
