import os
import json
import re
from PIL import Image
import torch
from datetime import datetime
import traceback
from typing import Tuple
from huggingface_hub import snapshot_download
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)


def _ensure_hf_caches_on_windows():
    """Set HF cache envs to safe paths (avoid symlinks issues on Windows)."""
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "hf_home")
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "hf_home", "hub")
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "hf_home", "transformers")


def _download_snapshot(model_id: str, local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=2,
    )
    return local_dir


def load_qwen_vl(
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    cache_subdir: str = "qwen2-vl-7b-instruct",
) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    _ensure_hf_caches_on_windows()

    local_dir = os.path.join(os.environ["TRANSFORMERS_CACHE"], cache_subdir)
    _download_snapshot(model_id=model_id, local_dir=local_dir)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        local_dir,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        local_dir,
        trust_remote_code=True
    )
    return model, processor


def _strip_examples_from_prompt(prompt_text):
    marker = "Final Output"
    if marker in prompt_text:
        parts = prompt_text.split(marker, 1)
        return parts[0] + marker + "\nYour final output should be a single, clean JSON object."
    return prompt_text


def generate_sound_json(model, processor, image_path, prompt, use_few_shot=True, example_images=None):
    try:
        core_instruction = _strip_examples_from_prompt(prompt)
        
        messages = []
        
        if use_few_shot and example_images:
            # Few-shot 예시들 추가
            for ex_img_path, ex_json in example_images:
                if os.path.exists(ex_img_path):
                    messages.extend([
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image", "image": ex_img_path},
                                {"type": "text", "text": "Analyze this image and generate a sound source JSON."}
                            ]
                        },
                        {"role": "assistant", "content": ex_json}
                    ])
        
        # 현재 처리할 이미지 추가
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": core_instruction + " Output only the JSON object."}
            ]
        })
        
        # 텍스트 생성
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 이미지들 수집 및 로드
        image_inputs = []
        for message in messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content.get("type") == "image":
                        img_path = content["image"]
                        if os.path.exists(img_path):
                            img = Image.open(img_path).convert('RGB')
                            image_inputs.append(img)
        
        print(f"처리 중인 이미지들: {len(image_inputs)}개")
        
        # 프로세서 호출
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
            )
        
        # 디코딩
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def parse_json_response(response):
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_json = json.loads(json_str)
            return {
                "success": True,
                "json_data": parsed_json,
                "raw_response": response
            }
        else:
            return {
                "success": False,
                "json_data": None,
                "raw_response": response,
                "error": "No JSON found in response"
            }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "json_data": None,
            "raw_response": response,
            "error": f"JSON parsing error: {str(e)}"
        }


def process_image_with_vlm(model, processor, image_path, prompt, example_images=None):
    """VLM을 사용하여 이미지를 처리하고 JSON 결과를 반환"""
    response = generate_sound_json(model, processor, image_path, prompt, use_few_shot=True, example_images=example_images)
    parsed = parse_json_response(response)
    return parsed