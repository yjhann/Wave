#!/usr/bin/env python3
"""
Scene-to-Sound Generation Pipeline
이미지에서 사운드 소스를 추출하고 오디오를 생성하는 통합 파이프라인
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

from image_to_text import batch_process_images, process_single_image_with_vlm
from audioldm2 import run_generation as generate_audio
from utils import check_required_directories, check_required_files, ensure_dir


def print_banner():
    """프로그램 시작 배너 출력"""
    print("=" * 80)
    print("🎵 Scene-to-Sound Generation Pipeline")
    print("=" * 80)
    print("📸 이미지 → 🧠 VLM → 🎯 Sound Sources → 🎵 Audio Generation")
    print("=" * 80)


def print_step(step: int, total: int, description: str):
    """단계별 진행 상황 출력"""
    print(f"\n[{step}/{total}] {description}")
    print("-" * 60)


def check_dependencies():
    """필수 디렉토리와 파일들 확인"""
    print("🔍 의존성 확인 중...")
    
    # 필수 디렉토리 확인
    required_dirs = ["data", "vlm_prompt"]
    missing_dirs = check_required_directories(required_dirs)
    
    if missing_dirs:
        print(f"❌ 필수 디렉토리가 없습니다: {', '.join(missing_dirs)}")
        return False
    
    # vlm_prompt 폴더 내 필수 파일 확인
    required_files = [
        "vlm_prompt/111_sound_source.json",
        "vlm_prompt/211_sound_source.json",
        "vlm_prompt/image/111.jpg",
        "vlm_prompt/image/211.jpg"
    ]
    
    missing_files = check_required_files(required_files)
    
    if missing_files:
        print(f"❌ 필수 파일이 없습니다: {', '.join(missing_files)}")
        return False
    
    # data 폴더에 이미지 파일 확인
    from utils import find_image_files
    data_images = find_image_files("data")
    
    if not data_images:
        print("❌ data 폴더에 이미지 파일이 없습니다")
        return False
    
    print(f"✅ 의존성 확인 완료! (data 폴더: {len(data_images)}개 이미지)")
    return True


def run_full_pipeline(
    data_dir: str = "data",
    sound_sources_dir: str = "sound_sources", 
    result_dir: str = "result",
    single_image: Optional[str] = None,
    skip_vlm: bool = False,
    skip_audio: bool = False,
    audio_model: str = "cvssp/audioldm-s-full-v2",
    audio_seconds: float = 4.0,
    audio_steps: int = 200,
    audio_guidance: float = 3.5,
    audio_seed: Optional[int] = None
) -> Dict[str, Any]:
    """전체 파이프라인 실행"""
    
    results = {
        "start_time": datetime.now().isoformat(),
        "steps_completed": [],
        "errors": []
    }
    
    try:
        # 1단계: 의존성 확인
        print_step(1, 4, "의존성 확인")
        if not check_dependencies():
            results["errors"].append("의존성 확인 실패")
            return results
        
        results["steps_completed"].append("dependency_check")
        
        # 2단계: VLM을 사용한 Sound Sources 생성
        if not skip_vlm:
            print_step(2, 4, "VLM을 사용한 Sound Sources 생성")
            
            if single_image:
                # 단일 이미지 처리
                print(f"단일 이미지 처리: {single_image}")
                result = process_single_image_with_vlm(single_image, sound_sources_dir)
                
                if result.get("success"):
                    print(f"✅ 단일 이미지 처리 완료: {result.get('output_json_path')}")
                    results["steps_completed"].append("vlm_single")
                else:
                    print(f"❌ 단일 이미지 처리 실패: {result.get('error')}")
                    results["errors"].append(f"VLM 단일 처리 실패: {result.get('error')}")
            else:
                # 배치 처리
                vlm_results = batch_process_images(data_dir, sound_sources_dir)
                
                if vlm_results and not vlm_results.get("error"):
                    successful = len(vlm_results.get("successful_results", []))
                    total = vlm_results.get("summary", {}).get("processing_info", {}).get("total_images", 0)
                    print(f"✅ VLM 배치 처리 완료: {successful}/{total} 성공")
                    results["steps_completed"].append("vlm_batch")
                    results["vlm_results"] = vlm_results
                else:
                    print("❌ VLM 배치 처리 실패")
                    results["errors"].append("VLM 배치 처리 실패")
        else:
            print("⏭️ VLM 단계 건너뛰기")
            results["steps_completed"].append("vlm_skipped")
        
        # 3단계: Sound Sources 확인
        print_step(3, 4, "Sound Sources 확인")
        
        sound_source_files = []
        if os.path.exists(sound_sources_dir):
            for root, dirs, files in os.walk(sound_sources_dir):
                for file in files:
                    if file.endswith('_sound_source.json'):
                        sound_source_files.append(os.path.join(root, file))
        
        if not sound_source_files:
            print("❌ Sound Sources JSON 파일을 찾을 수 없습니다")
            results["errors"].append("Sound Sources 파일 없음")
            return results
        
        print(f"✅ {len(sound_source_files)}개 Sound Sources 파일 발견")
        results["sound_source_files"] = sound_source_files
        
        # 4단계: AudioLDM2를 사용한 오디오 생성
        if not skip_audio:
            print_step(4, 4, "AudioLDM2를 사용한 오디오 생성")
            
            try:
                generate_audio(
                    sound_source_dir=sound_sources_dir,
                    result_dir=result_dir,
                    model_id=audio_model,
                    audio_seconds=audio_seconds,
                    steps=audio_steps,
                    guidance=audio_guidance,
                    seed=audio_seed,
                    single=single_image
                )
                print("✅ 오디오 생성 완료")
                results["steps_completed"].append("audio_generation")
            except Exception as e:
                print(f"❌ 오디오 생성 실패: {str(e)}")
                results["errors"].append(f"오디오 생성 실패: {str(e)}")
        else:
            print("⏭️ 오디오 생성 단계 건너뛰기")
            results["steps_completed"].append("audio_skipped")
        
        # 완료 메시지
        print("\n" + "=" * 80)
        print("🎉 파이프라인 실행 완료!")
        print("=" * 80)
        
        if results["steps_completed"]:
            print("✅ 완료된 단계:")
            for step in results["steps_completed"]:
                print(f"  - {step}")
        
        if results["errors"]:
            print("❌ 오류 발생:")
            for error in results["errors"]:
                print(f"  - {error}")
        
        print(f"📁 Sound Sources: {sound_sources_dir}")
        print(f"📁 결과 오디오: {result_dir}")
        
    except Exception as e:
        print(f"💥 파이프라인 실행 중 오류 발생: {str(e)}")
        results["errors"].append(f"파이프라인 오류: {str(e)}")
        import traceback
        traceback.print_exc()
    
    results["end_time"] = datetime.now().isoformat()
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Scene-to-Sound Generation Pipeline")
    
    # 기본 설정
    parser.add_argument("--data_dir", type=str, default="data", help="입력 이미지 디렉토리")
    parser.add_argument("--sound_sources_dir", type=str, default="sound_sources", help="Sound sources 출력 디렉토리")
    parser.add_argument("--result_dir", type=str, default="result", help="최종 오디오 출력 디렉토리")
    
    # 단일 이미지 처리
    parser.add_argument("--single", type=str, default=None, help="단일 이미지 파일 경로")
    
    # 단계별 스킵 옵션
    parser.add_argument("--skip_vlm", action="store_true", help="VLM 단계 건너뛰기")
    parser.add_argument("--skip_audio", action="store_true", help="오디오 생성 단계 건너뛰기")
    
    # 오디오 생성 설정
    parser.add_argument("--audio_model", type=str, default="cvssp/audioldm-s-full-v2", help="AudioLDM 모델 ID")
    parser.add_argument("--audio_seconds", type=float, default=4.0, help="오디오 길이 (초)")
    parser.add_argument("--audio_steps", type=int, default=200, help="Diffusion 스텝 수")
    parser.add_argument("--audio_guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--audio_seed", type=int, default=None, help="랜덤 시드")
    
    # 결과 저장
    parser.add_argument("--save_log", type=str, default=None, help="실행 로그 저장 파일")
    
    args = parser.parse_args()
    
    print_banner()
    
    # 결과 디렉토리 생성
    ensure_dir(args.sound_sources_dir)
    ensure_dir(args.result_dir)
    
    # 파이프라인 실행
    results = run_full_pipeline(
        data_dir=args.data_dir,
        sound_sources_dir=args.sound_sources_dir,
        result_dir=args.result_dir,
        single_image=args.single,
        skip_vlm=args.skip_vlm,
        skip_audio=args.skip_audio,
        audio_model=args.audio_model,
        audio_seconds=args.audio_seconds,
        audio_steps=args.audio_steps,
        audio_guidance=args.audio_guidance,
        audio_seed=args.audio_seed
    )
    
    # 로그 저장
    if args.save_log:
        with open(args.save_log, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📄 실행 로그 저장: {args.save_log}")


if __name__ == "__main__":
    main()
