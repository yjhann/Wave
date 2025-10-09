# WAVE
25-2 DSL Multi-modal modeling project

====================================

이 프로젝트는 이미지에서 사운드 소스를 추출하고 오디오를 생성하는 통합 파이프라인입니다.

파일 구조
---------
- main.py: 전체 파이프라인 통합 실행 파일
- image_to_text.py: VLM을 사용한 이미지에서 sound sources 추출
- vlm_qwen.py: Qwen2-VL 모델 로딩 및 처리
- audio_prompt.py: VLM 출력을 AudioLDM2용 프롬프트로 변환
- audioldm2.py: AudioLDM2를 사용한 오디오 생성
- vlm_prompt/extract_sources.py: 프롬프트 생성 및 예시 데이터 로드
- data/: 입력 이미지 폴더
- vlm_prompt/: VLM 프롬프트 및 예시 데이터
- sound_sources/: VLM 출력 JSON 저장 폴더
- result/: 최종 오디오 파일 저장 폴더

사용법
------
1. 전체 파이프라인 실행:
   python main.py

2. 단일 이미지 처리:
   python main.py --single data/101.jpg

3. VLM 단계만 실행:
   python main.py --skip_audio

4. 오디오 생성만 실행:
   python main.py --skip_vlm

5. 배치 처리 (data 폴더의 모든 이미지):
   python image_to_text.py

6. 오디오 생성:
   python audioldm2.py

필수 요구사항
-------------
- Python 3.8+
- CUDA 지원 GPU (권장)
- 충분한 메모리 (최소 8GB RAM, 16GB+ 권장)

설치
----
pip install -r requirements.txt

주의사항
--------
- 첫 실행 시 모델 다운로드로 인해 시간이 오래 걸릴 수 있습니다
- GPU 메모리 부족 시 오디오 길이나 배치 크기를 조정하세요
- Hugging Face 토큰이 필요한 경우 환경변수로 설정하세요:
  export HUGGING_FACE_TOKEN=your_token_here
