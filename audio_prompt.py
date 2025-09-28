import json
import os
from typing import Dict, List, Any


def generate_prompts(data: dict, custom_templates: dict = None) -> List[Dict[str, Any]]:
    """VLM 출력 JSON을 AudioLDM2용 프롬프트로 변환"""
    if custom_templates is None:
        custom_templates = {}

    scene = data.get("scene_description", "A neutral scene")
    mood = data.get("mood_description", "a neutral mood")
    
    generated_prompts = []

    for source in data.get("sound_sources", []):
        name = source.get("name", "an object")
        material = source.get("material", "unknown material")

        for variant in source.get("variants", []):
            play_method = variant.get("play_method")
            timbre = variant.get("timbre", [])
            instrument = variant.get("mapping_to_music_instrument")
            timbre_str = ", ".join(timbre)
            
            core_prompt = ""
            
            if play_method in custom_templates:
                template = custom_templates[play_method]
                core_prompt = template.format(
                    name=name, material=material, play_method=play_method, 
                    timbre_str=timbre_str, instrument=instrument
                )
            elif instrument and instrument.lower() != "none":
                core_prompt = (
                    f"Generate a high-fidelity, realistic sound effect.\n"
                    f"The sound source is '{name}' made of '{material}'.\n"
                    f"The action is a '{play_method}', creating a sound with a '{timbre_str}' timbre.\n"
                    f"For this, use the sonic character of a '{instrument}' as an inspirational reference for the sound's quality, "
                    f"especially its '{timbre_str}' aspects.\n"
                    f"The final audio must be a completely natural sound, not a musical note."
                )
            else:
                core_prompt = (
                    f"Generate a high-fidelity, realistic sound effect.\n"
                    f"The sound source is '{name}' made of '{material}'.\n"
                    f"The action is a '{play_method}', creating a sound with a '{timbre_str}' timbre.\n"
                    f"The recording should be clean and detailed, sounding authentic as if captured in a real-world environment. "
                    f"Focus on realism, not musicality."
                )

            final_prompt = (
                f"Context: The scene is '{scene}'. The overall mood is '{mood}'.\n\n"
                f"{core_prompt}\n\n"
                f"Crucially, the generated sound must be consistent with the '{mood}' mood and not feel out of place."
            )
            
            generated_prompts.append({
                "source_name": name,
                "play_method": play_method,
                "prompt": final_prompt
            })
            
    return generated_prompts


def process_sound_sources_json(json_path: str) -> List[Dict[str, Any]]:
    """JSON 파일을 읽어서 프롬프트 생성"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = generate_prompts(data)
        return prompts
    except Exception as e:
        print(f"JSON 파일 처리 중 오류 발생: {str(e)}")
        return []


def batch_process_sound_sources(sound_sources_dir: str = "sound_sources") -> Dict[str, List[Dict[str, Any]]]:
    """sound_sources 디렉토리의 모든 JSON 파일을 처리하여 프롬프트 생성"""
    results = {}
    
    if not os.path.exists(sound_sources_dir):
        print(f"Sound sources 디렉토리가 존재하지 않습니다: {sound_sources_dir}")
        return results
    
    # 각 이미지별 폴더에서 JSON 파일 찾기
    for image_folder in os.listdir(sound_sources_dir):
        image_folder_path = os.path.join(sound_sources_dir, image_folder)
        if not os.path.isdir(image_folder_path):
            continue
            
        # 해당 폴더에서 JSON 파일 찾기
        json_files = [f for f in os.listdir(image_folder_path) if f.endswith('_sound_source.json')]
        
        for json_file in json_files:
            json_path = os.path.join(image_folder_path, json_file)
            print(f"처리 중: {json_path}")
            
            prompts = process_sound_sources_json(json_path)
            if prompts:
                results[json_path] = prompts
                print(f"  ✅ {len(prompts)}개 프롬프트 생성")
            else:
                print(f"  ❌ 프롬프트 생성 실패")
    
    return results


def save_prompts_to_file(prompts: List[Dict[str, Any]], output_path: str) -> None:
    """생성된 프롬프트를 파일로 저장"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        print(f"프롬프트 저장 완료: {output_path}")
    except Exception as e:
        print(f"프롬프트 저장 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM 출력을 AudioLDM2용 프롬프트로 변환")
    parser.add_argument("--input", type=str, help="입력 JSON 파일 경로")
    parser.add_argument("--output", type=str, help="출력 프롬프트 파일 경로")
    parser.add_argument("--batch", action="store_true", help="배치 처리 모드")
    parser.add_argument("--sound_sources_dir", type=str, default="sound_sources", help="Sound sources 디렉토리")
    
    args = parser.parse_args()
    
    if args.batch:
        print("배치 처리 모드")
        results = batch_process_sound_sources(args.sound_sources_dir)
        print(f"총 {len(results)}개 파일 처리 완료")
    elif args.input:
        print(f"단일 파일 처리: {args.input}")
        prompts = process_sound_sources_json(args.input)
        if prompts:
            if args.output:
                save_prompts_to_file(prompts, args.output)
            else:
                print("생성된 프롬프트:")
                for i, prompt in enumerate(prompts, 1):
                    print(f"\n{i}. {prompt['source_name']} - {prompt['play_method']}")
                    print(f"   {prompt['prompt'][:100]}...")
        else:
            print("프롬프트 생성 실패")
    else:
        print("사용법:")
        print("  python audio_prompt.py --input <json_file> [--output <output_file>]")
        print("  python audio_prompt.py --batch [--sound_sources_dir <dir>]")