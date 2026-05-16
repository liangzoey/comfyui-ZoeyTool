# -*- coding: utf-8 -*-
"""
Zoey Video Nodes - 视频处理专用节点
包含：批量加载、批处理、智能存储
"""

import os
import torch
import cv2
import glob
import re
import shutil
import time
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from comfy.utils import ProgressBar

# ====================== 节点 1: 批量视频加载器 ======================
class BatchVideoLoader:
    """批量视频加载器 (修复排序问题)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "D:/Videos/Input", "folder_meta": True}),
                "file_pattern": ("STRING", {"default": "*.mp4;*.avi;*.mov"}),
                "limit": ("INT", {"default": 20, "min": 1})
            },
            "optional": {
                "refresh": ("BOOLEAN", {"default": False, "label_on": "刷新", "label_off": "默认"})
            }
        }
    RETURN_TYPES = ("VIDEO_BATCH", "STRING")
    RETURN_NAMES = ("视频列表", "源目录")
    FUNCTION = "load_batch"
    CATEGORY = "Zoey Tool/视频处理"

    def load_batch(self, directory: str, file_pattern: str, limit: int, refresh: bool):
        directory = os.path.normpath(directory)
        if not os.path.isdir(directory):
            raise ValueError(f"目录不存在: {directory}")
        
        video_files = []
        for pattern in file_pattern.split(';'):
            pattern = pattern.strip()
            if pattern:
                full_pattern = os.path.join(directory, pattern)
                video_files.extend(glob.glob(full_pattern, recursive=False))
        
        # 自然排序
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        video_files.sort(key=natural_sort_key)
        video_files = video_files[:max(1, limit)]
        return (video_files, directory)

# ====================== 节点 2: 视频批处理器 ======================
class VideoBatchProcessor:
    """视频批处理器 (修复颜色问题)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("VIDEO_BATCH",),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 60}),
                "output_format": (["mp4", "avi", "mov"], {"default": "mp4"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "skip_existing": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"})
            }
        }
    RETURN_TYPES = ("VIDEO_BATCH",)
    RETURN_NAMES = ("处理视频",)
    FUNCTION = "process_batch"
    CATEGORY = "Zoey Tool/视频处理"

    def process_batch(self, videos: List[str], frame_rate: int, output_format: str, device: str, skip_existing: bool = True) -> Tuple[List[Dict[str, str]]]:
        self._check_pyav_dependency()
        device = self._select_device(device)
        processed_videos = []
        pbar = ProgressBar(len(videos))
        
        for i, video_path in enumerate(videos):
            try:
                output_dir = os.path.join(os.path.dirname(video_path), "processed")
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
                
                if skip_existing and os.path.exists(output_path):
                    continue
                    
                self._process_video(input_path=video_path, output_path=output_path, frame_rate=frame_rate, output_format=output_format)
                
                processed_videos.append({
                    "source": video_path,
                    "output": output_path,
                    "format": output_format
                })
                pbar.update(1)
            except Exception as e:
                continue
        return (processed_videos,)

    def _check_pyav_dependency(self):
        try:
            import av
            if av.__version__ < "12.0.0":
                raise ImportError("PyAV版本过低")
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "av", "--upgrade", "--no-cache-dir"], check=True)

    def _select_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _process_video(self, input_path: str, output_path: str, frame_rate: int, output_format: str):
        if output_format == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_format == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else: # mov
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = frame_rate if frame_rate > 0 else orig_fps
        
        out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=target_fps, frameSize=(width, height))
        if not out.isOpened():
            raise RuntimeError(f"无法初始化视频写入器: {output_path}")

        # 修复：直接写入原始帧（避免颜色转换）
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame) # 直接写入原始帧
        cap.release()
        out.release()

# ====================== 节点 3: 智能视频存储器 ======================
class EnhancedVideoSaver:
    """智能视频存储器"""
    batch_counter = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_videos": ("VIDEO_BATCH",),
                "output_directory": ("STRING", {"default": "D:/视频输出", "folder_meta": True}),
                "prefix": ("STRING", {"default": "batch"}),
            },
            "optional": {
                "metadata": ("STRING", {"multiline": True, "default": ""}),
                "start_index": ("INT", {"default": 1, "min": 1, "step": 1}),
                "digits": ("INT", {"default": 4, "min": 1, "max": 6}),
                "date_folder": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"})
            }
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_batch"
    CATEGORY = "Zoey Tool/视频处理"

    def save_batch(self, processed_videos: list, output_directory: str, prefix: str, metadata="", start_index=1, digits=4, date_folder=True):
        output_directory = os.path.normpath(output_directory)
        os.makedirs(output_directory, exist_ok=True)
        
        # 按日期创建子目录
        if date_folder:
            date_str = datetime.now().strftime("%Y%m%d")
            output_directory = os.path.join(output_directory, date_str)
            os.makedirs(output_directory, exist_ok=True)
            
        self.batch_counter = 0 # 重置计数器
        current_index = max(start_index - 1, self.batch_counter)
        save_count = 0
        
        for idx, video in enumerate(processed_videos):
            try:
                index_str = str(current_index + idx + 1).zfill(digits)
                base_name = f"{prefix}_{index_str}"
                video_filename = f"{base_name}.{video['format']}"
                video_path = os.path.join(output_directory, video_filename)
                txt_path = os.path.join(output_directory, f"{base_name}_metadata.txt")
                
                # 复制视频文件
                if os.path.exists(video["output"]):
                    shutil.copy2(video["output"], video_path)
                else:
                    raise FileNotFoundError(f"源文件不存在: {video['output']}")
                
                # 写入元数据
                if metadata:
                    self._write_metadata(txt_path, metadata, base_name, video)
                save_count += 1
            except Exception as e:
                continue
        self.batch_counter += len(processed_videos)
        return ()

    def _write_metadata(self, path: str, metadata: str, base_name: str, video: dict):
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== 视频处理元数据 ===\n")
            f.write(f"文件名: {base_name}\n")
            f.write(f"格式: {video['format']}\n")
            f.write(f"源路径: {video['source']}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n自定义元数据:\n")
            f.write(metadata)

# ====================== 注册 ======================
NODE_CLASS_MAPPINGS = {
    "BatchVideoLoader": BatchVideoLoader,
    "VideoBatchProcessor": VideoBatchProcessor,
    "EnhancedVideoSaver": EnhancedVideoSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchVideoLoader": "Zoey - 批量视频加载器",
    "VideoBatchProcessor": "Zoey - 视频批处理器",
    "EnhancedVideoSaver": "Zoey - 智能视频存储器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']