import json
import os

def load_example_data():
    """vlm_prompt 폴더에서 예시 데이터를 로드"""
    example_images = []
    
    # 111번 예시
    ex1_img = os.path.join("vlm_prompt", "image", "111.jpg")
    ex1_json_path = os.path.join("vlm_prompt", "111_sound_source.json")
    if os.path.exists(ex1_img) and os.path.exists(ex1_json_path):
        with open(ex1_json_path, 'r', encoding='utf-8') as f:
            ex1_json = json.load(f)
        example_images.append((ex1_img, json.dumps(ex1_json, ensure_ascii=False)))
    
    # 211번 예시
    ex2_img = os.path.join("vlm_prompt", "image", "211.jpg")
    ex2_json_path = os.path.join("vlm_prompt", "211_sound_source.json")
    if os.path.exists(ex2_img) and os.path.exists(ex2_json_path):
        with open(ex2_json_path, 'r', encoding='utf-8') as f:
            ex2_json = json.load(f)
        example_images.append((ex2_img, json.dumps(ex2_json, ensure_ascii=False)))
    
    return example_images

def get_scene_to_sound_prompt():
    """audio_prompt.py의 함수를 포함한 프롬프트 생성"""
    from audio_prompt import generate_prompts
    
    # audio_prompt.py의 함수를 문자열로 변환
    audio_prompt_func = f"""
def generate_prompts(data: dict, custom_templates: dict = None) -> list[dict]:
    if custom_templates is None:
        custom_templates = {{}}

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
                    f"Generate a high-fidelity, realistic sound effect.\\n"
                    f"The sound source is '{{name}}' made of '{{material}}'.\\n"
                    f"The action is a '{{play_method}}', creating a sound with a '{{timbre_str}}' timbre.\\n"
                    f"For this, use the sonic character of a '{{instrument}}' as an inspirational reference for the sound's quality, "
                    f"especially its '{{timbre_str}}' aspects.\\n"
                    f"The final audio must be a completely natural sound, not a musical note."
                )
            else:
                core_prompt = (
                    f"Generate a high-fidelity, realistic sound effect.\\n"
                    f"The sound source is '{{name}}' made of '{{material}}'.\\n"
                    f"The action is a '{{play_method}}', creating a sound with a '{{timbre_str}}' timbre.\\n"
                    f"The recording should be clean and detailed, sounding authentic as if captured in a real-world environment. "
                    f"Focus on realism, not musicality."
                )

            final_prompt = (
                f"Context: The scene is '{{scene}}'. The overall mood is '{{mood}}'.\\n\\n"
                f"{{core_prompt}}\\n\\n"
                f"Crucially, the generated sound must be consistent with the '{{mood}}' mood and not feel out of place."
            )
            
            generated_prompts.append({{
                "source_name": name,
                "play_method": play_method,
                "prompt": final_prompt
            }})
            
    return generated_prompts
"""
    
    # 예시 데이터 로드
    example_images = load_example_data()
    ex1_json = ""
    ex2_json = ""
    
    if len(example_images) >= 1:
        ex1_json = example_images[0][1]
    if len(example_images) >= 2:
        ex2_json = example_images[1][1]
    
    prompt = f"""
You are an expert Scene-to-Sound Data Architect. Your mission is to analyze the provided image and generate a single, structured JSON output describing its potential soundscape. This JSON will be used as input for an audio generation model through the following functions:

{audio_prompt_func}

To achieve this, follow these steps in your internal thought process. Do not output the results of each step; only provide the final, complete JSON object.

Step 1: Analyze the Scene
First, mentally perform a thorough analysis of the image:

Summarize the Scene: Formulate a single, vivid sentence that captures the essence of the scene. This will become the scene_description.

Identify the Mood: Determine the overall atmosphere and list relevant keywords. This will be the mood_description.

List Objects & Materials: Identify all key objects and infer their primary materials.

Step 2: Brainstorm Potential Sound Sources
Based on your analysis, brainstorm a comprehensive list of potential sound effects (variants) that are consistent with the scene and mood.

Consider Interactions: Think about sounds from object-object interactions (e.g., footsteps on a surface) and object-environment interactions (e.g., leaves rustling in an unseen wind).

Adhere to Constraints: Do not include human speech or emotional sounds (like laughter or crying). Physical sounds like footsteps, clapping, or rustling clothes are acceptable.

Define Sound Properties: For each brainstormed sound, determine its:

play_method: The action causing the sound (e.g., rustle, knock, footstep, drip).

timbre: Descriptive keywords for the sound's character (e.g., "natural, organic").

mapping_to_music_instrument: Map the sound's character to guitar, bass, keyboard, or drums. If no clear mapping exists, you must use None.

Step 3: Structure, Score, and Filter the Final Output
Now, assemble your findings into the final JSON structure with the following rules:

Group Variants: Group all brainstormed sound ideas as a list of variants under their parent object.

Assign Confidence: For each variant, add a confidence score (from 0.0 to 1.0) representing how realistically and appropriately the sound fits the scene, mood, and material.

Filter by Confidence: Discard any variant with a confidence score below 0.7.

Ensure Minimum Count: This is a critical rule. The final JSON must contain at least 5 total variants across all objects. If your filtering results in fewer than 5, you must re-evaluate the confidence scores of the most plausible discarded ideas and include them to meet the minimum requirement.

Final Output
Your final output should be a single, clean JSON object that strictly adheres to the structure shown in the example below.

JSON Output Example 1:
{ex1_json}

JSON Output Example 2:
{ex2_json}
"""
    
    return prompt, example_images
