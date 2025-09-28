import os
import json
import glob
from datetime import datetime
import traceback
from typing import List, Dict, Any

from vlm_qwen import load_qwen_vl, process_image_with_vlm
from vlm_prompt.extract_sources import get_scene_to_sound_prompt
from utils import find_image_files, ensure_dir


def find_images_in_data_folder(data_dir: str = "data") -> List[str]:
    """data 폴더에서 이미지 파일들을 찾아 반환"""
    image_files = find_image_files(data_dir)
    
    if not image_files:
        print(f"Data 폴더에 이미지 파일이 없습니다: {data_dir}")
    
    return image_files


def process_single_image(model, processor, image_path: str, output_dir: str, prompt: str, example_images: List) -> Dict[str, Any]:
    """단일 이미지를 처리하여 sound source JSON 생성"""
    print(f"\n처리 중: {os.path.basename(image_path)}")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        print("JSON 생성 중...")
        
        # VLM을 사용하여 이미지 처리
        parsed = process_image_with_vlm(model, processor, image_path, prompt, example_images)
        
        result = {
            "image_path": image_path,
            "filename": base_name,
            "success": parsed['success'],
            "timestamp": datetime.now().isoformat()
        }
        
        if parsed['success']:
            json_data = parsed['json_data']
            
            # JSON 구조 검증
            validation_issues = validate_json_structure(json_data)
            total_variants = count_total_variants(json_data)
            
            result.update({
                "json_data": json_data,
                "total_variants": total_variants,
                "validation_issues": validation_issues,
                "meets_minimum_variants": total_variants >= 5,
                "sound_sources_count": len(json_data.get('sound_sources', [])),
                "scene_description": json_data.get('scene_description', 'N/A'),
                "mood_description": json_data.get('mood_description', 'N/A')
            })
            
            # 이미지별 폴더 생성
            image_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(image_output_dir, exist_ok=True)
            
            # JSON 파일 저장
            json_filename = f"{base_name}_sound_source.json"
            json_path = os.path.join(image_output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            result["output_json_path"] = json_path
            
            print(f"✅ 성공! Variants: {total_variants}/5")
            if total_variants < 5:
                print(f"⚠️ 최소 요구사항 미달 (5개 미만)")
            if validation_issues:
                print(f"⚠️ 구조 문제: {len(validation_issues)}개")
                
        else:
            result.update({
                "error": parsed['error'],
                "raw_response": parsed['raw_response']
            })
            print(f"❌ 실패: {parsed['error']}")
            
    except Exception as e:
        result.update({
            "success": False,
            "error": f"Processing error: {str(e)}",
            "traceback": traceback.format_exc()
        })
        print(f"💥 오류: {str(e)}")
    
    return result


def count_total_variants(json_data: Dict[str, Any]) -> int:
    """JSON 데이터에서 총 variants 수 계산"""
    if not isinstance(json_data, dict):
        return 0
    total_variants = 0
    for source in json_data.get('sound_sources', []):
        total_variants += len(source.get('variants', []))
    return total_variants


def validate_json_structure(json_data: Dict[str, Any]) -> List[str]:
    """JSON 구조 검증"""
    required_fields = ['scene_description', 'mood_description', 'sound_sources']
    issues = []
    
    for field in required_fields:
        if field not in json_data:
            issues.append(f"Missing field: {field}")
    
    if 'sound_sources' in json_data:
        sources = json_data['sound_sources']
        if not isinstance(sources, list):
            issues.append("sound_sources should be a list")
        else:
            for i, source in enumerate(sources):
                source_required = ['name', 'material', 'variants']
                for field in source_required:
                    if field not in source:
                        issues.append(f"sound_sources[{i}] missing field: {field}")
                
                if 'variants' in source:
                    for j, variant in enumerate(source['variants']):
                        variant_required = ['play_method', 'timbre', 'mapping_to_music_instrument']
                        for field in variant_required:
                            if field not in variant:
                                issues.append(f"sound_sources[{i}].variants[{j}] missing field: {field}")
    
    return issues


def batch_process_images(data_dir: str = "data", output_dir: str = "sound_sources") -> Dict[str, Any]:
    """data 폴더의 모든 이미지를 배치 처리"""
    print("🚀 배치 Sound Source 생성 시작")
    print("=" * 80)

    ensure_dir(output_dir)
    print(f"출력 디렉토리: {output_dir}")
    
    # VLM 모델 로드
    print("VLM 모델 로딩 중...")
    model, processor = load_qwen_vl()
    print(f"✅ VLM 모델 로드 완료! Device: {model.device}")
    
    # 프롬프트 및 예시 데이터 로드
    prompt, example_images = get_scene_to_sound_prompt()
    print(f"✅ 프롬프트 로드 완료! 예시 이미지: {len(example_images)}개")
  
    # data 폴더에서 이미지 파일 찾기
    image_files = find_images_in_data_folder(data_dir)
    
    if not image_files:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {data_dir}")
        return {"error": "No images found"}
    
    print(f"📸 총 {len(image_files)}개 이미지 파일 발견")
    print("처리할 파일 목록:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    # 결과 저장용
    all_results = []
    successful_results = []
    failed_results = []
    insufficient_variants = []
    
    # 각 이미지 처리
    print("\n" + "=" * 80)
    print("이미지 처리 시작")
    print("=" * 80)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        
        result = process_single_image(model, processor, image_path, output_dir, prompt, example_images)
        all_results.append(result)
        
        if result['success']:
            successful_results.append(result)
            if not result['meets_minimum_variants']:
                insufficient_variants.append(result)
        else:
            failed_results.append(result)
    
    # 종합 결과 저장
    summary = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "insufficient_variants": len(insufficient_variants)
        },
        "results": all_results
    }
    
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 최종 리포트 출력
    print("\n" + "=" * 80)
    print("📊 최종 처리 결과")
    print("=" * 80)
    print(f"📸 총 이미지 수: {len(image_files)}")
    print(f"✅ 성공: {len(successful_results)}")
    print(f"❌ 실패: {len(failed_results)}")
    print(f"⚠️ Variants 부족 (5개 미만): {len(insufficient_variants)}")
    print(f"📁 요약 파일: {summary_path}")
    
    if insufficient_variants:
        print("\n🔍 Variants 부족 파일들:")
        for result in insufficient_variants:
            print(f"  - {result['filename']}: {result['total_variants']}개 variants")
    
    if failed_results:
        print("\n❌ 실패한 파일들:")
        for result in failed_results:
            print(f"  - {result['filename']}: {result.get('error', 'Unknown error')}")
    
    # 성공률 계산
    success_rate = (len(successful_results) / len(image_files)) * 100
    minimum_variants_rate = ((len(successful_results) - len(insufficient_variants)) / len(image_files)) * 100
    
    print(f"\n📈 성공률: {success_rate:.1f}%")
    print(f"📈 최소 요구사항 충족률: {minimum_variants_rate:.1f}%")
    
    return {
        "summary": summary,
        "successful_results": successful_results,
        "failed_results": failed_results,
        "insufficient_variants": insufficient_variants
    }


def process_single_image_with_vlm(image_path: str, output_dir: str) -> Dict[str, Any]:
    """단일 이미지를 VLM으로 처리하는 고수준 함수"""
    try:
        # VLM 모델 로드
        print("VLM 모델 로딩 중...")
        model, processor = load_qwen_vl()
        print(f"✅ VLM 모델 로드 완료! Device: {model.device}")
        
        # 프롬프트 및 예시 데이터 로드
        prompt, example_images = get_scene_to_sound_prompt()
        print(f"✅ 프롬프트 로드 완료! 예시 이미지: {len(example_images)}개")
        
        # 단일 이미지 처리
        result = process_single_image(model, processor, image_path, output_dir, prompt, example_images)
        return result
        
    except Exception as e:
        print(f"❌ 단일 이미지 처리 중 오류 발생: {str(e)}")
        print("상세 오류:")
        print(traceback.format_exc())
        return {
            "success": False,
            "error": f"Processing error: {str(e)}",
            "traceback": traceback.format_exc()
        }


def run_batch_processing():
    """배치 처리 실행"""
    try:
        results = batch_process_images()
        return results
    except Exception as e:
        print(f"❌ 배치 처리 중 오류 발생: {str(e)}")
        print("상세 오류:")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, default=None, help="단일 이미지 경로 (예: data/101.jpg)")
    parser.add_argument("--out", type=str, default="sound_sources", help="출력 디렉토리")
    parser.add_argument("--data", type=str, default="data", help="입력 데이터 디렉토리")
    args = parser.parse_args()

    print("🚀 Sound Source 생성기")
    print(f"ARGS: single={args.single}, out={args.out}, data={args.data}")

    if args.single:
        print("모드: 단일 이미지 처리")
        ensure_dir(args.out)
        
        res = process_single_image_with_vlm(args.single, args.out)
        if res.get("success"):
            print("\n🎉 단일 처리 완료!")
            print(f"JSON: {res.get('output_json_path')}")
        else:
            print("\n💥 단일 처리 실패!")
    else:
        print("모드: 배치 처리")
        # 배치 처리 실행
        results = run_batch_processing()
        if results:
            print("\n🎉 배치 처리 완료!")
            print("생성된 파일들을 'sound_sources' 디렉토리에서 확인하세요.")
        else:
            print("\n💥 배치 처리 실패!")