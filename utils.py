"""
공통 유틸리티 함수들
중복 코드를 제거하고 재사용 가능한 기능들을 제공
"""

import os
from typing import List


def ensure_dir(path: str) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(path, exist_ok=True)


def find_image_files(directory: str, valid_extensions: set = None) -> List[str]:
    """디렉토리에서 이미지 파일들을 찾아 반환"""
    if valid_extensions is None:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
    
    for root_dir, _, files in os.walk(directory):
        for name in files:
            base, ext = os.path.splitext(name)
            if ext in valid_extensions:
                image_files.append(os.path.join(root_dir, name))
    
    return sorted(image_files)


def check_required_directories(required_dirs: List[str]) -> List[str]:
    """필수 디렉토리들이 존재하는지 확인하고 누락된 것들을 반환"""
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    return missing_dirs


def check_required_files(required_files: List[str]) -> List[str]:
    """필수 파일들이 존재하는지 확인하고 누락된 것들을 반환"""
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files


def sanitize_filename(text: str, max_length: int = 120) -> str:
    """파일명에 사용할 수 없는 문자를 제거하고 길이를 제한"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, "_")
    
    # 공백을 언더스코어로 변경하고 길이 제한
    text = "_".join(text.split())[:max_length]
    return text
