import os
import torch
import numpy as np
import glob
import re
import shutil
import time
import cv2
import requests
import json
import hashlib
import subprocess
import sys
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.utils import ProgressBar
from transformers import pipeline
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# ======================
# Zoey Tool - 节点集合
# ======================

class WanPromptGenerator:
    """Wan2.2提示词生成器 - 根据用户选择自动生成影视级视频提示词"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "主体": ("STRING", {"default": "一位年轻女孩", "multiline": False}),
                "场景": ("STRING", {"default": "阳光下的田野", "multiline": False}),
                "动作": ("STRING", {"default": "轻轻抚弄野花", "multiline": False}),
                "光源类型": (["无", "日光", "人工光", "月光", "实用光", "火光", "荧光", "阴天光", "混合光", "晴天光"],),
                "光线类型": (["无", "柔光", "硬光", "顶光", "侧光", "背光", "底光", "边缘光", "剪影", "低对比度", "高对比度"],),
                "时间段": (["无", "白天", "夜晚", "黄昏", "日落", "黎明", "日出"],),
                "景别": (["无", "特写", "近景", "中景", "中近景", "中全景", "全景", "广角"],),
                "构图方式": (["无", "中心构图", "平衡构图", "右侧重构图", "左侧重构图", "对称构图", "短边构图"],),
                "镜头焦段": (["极", "中焦距", "广角", "长焦", "望远", "超广角-鱼眼"],),
                "机位角度": (["无", "过肩角度", "高角度", "低角度", "倾斜角度", "航拍"],),
                "镜头类型": (["无", "干净的单人镜头", "双人镜头", "三人镜头", "群像镜头", "定场镜头"],),
                "色调风格": (["无", "暖色调", "冷色调", "高饱和度", "低饱和度"],),
                "画面风格": (["无", "毛毡风格", "3D卡通", "像素风格", "木偶动画", "3D游戏", "黏土风格", 
                                "二次元", "水彩画", "黑白动画", "油画风格", "移轴摄影", "延时拍摄"],),
                "面部表情": (["无", "愤怒", "恐惧", "高兴", "悲伤", "惊讶"],),
                "主体动作": (["无", "跑步", "滑滑板", "踢足球", "网球", "乒乓球", "滑雪", 
                                  "篮球", "橄榄球", "顶碗舞", "侧手翻"],),
                "镜头运动": (["无", "镜头推进", "镜头拉远", "镜头向右移动", "镜头向左移动", 
                                   "手持镜头", "复合运镜", "跟随镜头", "环绕运镜"],),
            },
            "optional": {
                "主体细节": ("STRING", {"multiline": True, "default": "身着少数民族服饰的黑发苗族少女"}),
                "场景细节": ("STRING", {"multiline": True, "default": "高草丛生的田野，远处有模糊的树木轮廓"}),
                "动作细节": ("STRING", {"multiline": True, "default": "双腿交叉坐下，双手轻轻抚弄身旁的野花"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Zoey Tool/提示词"
    
    def generate_prompt(self, **kwargs):
        prompt_parts = []
        
        # 主体部分
        if kwargs.get("主体细节", ""):
            prompt_parts.append(f"{kwargs['主体']}, {kwargs['主体细节']}")
        else:
            prompt_parts.append(kwargs["主体"])
        
        # 场景部分
        if kwargs.get("场景细节", ""):
            prompt_parts.append(f"{kwargs['场景']}, {kwargs['场景细节']}")
        else:
            prompt_parts.append(kwargs["场景"])
        
        # 动作部分
        if kwargs.get("动作细节", ""):
            prompt_parts.append(f"{kwargs['动作']}, {kwargs['动作细节']}")
        else:
            prompt_parts.append(kwargs["动作"])
        
        # 添加美学控制部分
        aesthetics = []
        
        # 光源类型
        if kwargs["光源类型"] != "无":
            aesthetics.append(kwargs["光源类型"])
        
        # 光线类型
        if kwargs["光线类型"] != "无":
            aesthetics.append(kwargs["光线类型"])
        
        # 时间段
        if kwargs["时间段"] != "极无":
            aesthetics.append(kwargs["时间段"])
        
        # 景别
        if kwargs["景别"] != "无":
            aesthetics.append(kwargs["景别"])
        
        # 构图
        if kwargs["构图方式"] != "无":
            aesthetics.append(kwargs["构图方式"])
        
        # 镜头焦段
        if kwargs["镜头焦段"] != "无":
            aesthetics.append(kwargs["镜头焦段"])
        
        # 机位角度
        if kwargs["机位角度"] != "无":
            aesthetics.append(kwargs["机位角度"])
        
        # 镜头类型
        if kwargs["镜头类型"] != "无":
            aesthetics.append(kwargs["镜头类型"])
        
        # 色调
        if kwargs["色调风格"] != "无":
            aesthetics.append(kwargs["色调风格"])
        
        # 添加美学控制到提示词
        if aesthetics:
            prompt_parts.append(", ".join(aesthetics))
        
        # 风格化
        if kwargs["画面风格"] != "无":
            prompt_parts.append(kwargs["画面风格"])
        
        # 面部情绪
        if kwargs["面部表情"] != "无":
            prompt_parts.append(kwargs["面部表情"])
        
        # 运动控制
        if kwargs["主体动作"] != "无":
            prompt_parts.append(kwargs["主体动作"])
        
        # 高级运镜
        if kwargs["镜头运动"] != "无":
            prompt_parts.append(kwargs["镜头运动"])
        
        # 组合所有部分
        full_prompt = ", ".join(prompt_parts)
        
        return (full_prompt,)

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
        
        # 自然排序功能
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', s)]
        video_files.sort(key=natural_sort_key)
        
        video_files = video_files[:max(1, limit)]
        return (video_files, directory)

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
    
    def process_batch(self, videos: List[str], frame_rate: int, output_format: str, 
                     device: str, skip_existing: bool = True) -> Tuple[List[Dict[str, str]]]:
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
                
                self._process_video(
                    input_path=video_path,
                    output_path=output_path,
                    frame_rate=frame_rate,
                    output_format=output_format
                )
                
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
            subprocess.run([
                sys.executable, "-m", "pip", 
                "install", "av", 
                "--upgrade", "--no-cache-dir"
            ], check=True)
    
    def _select_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _process_video(self, input_path: str, output_path: str, frame_rate: int, output_format: str):
        if output_format == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_format == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:  # mov
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = frame_rate if frame_rate > 0 else orig_fps
        
        out = cv2.VideoWriter(
            filename=output_path,
            fourcc=fourcc,
            fps=target_fps,
            frameSize=(width, height)
        )
        
        if not out.isOpened():
            raise RuntimeError(f"无法初始化视频写入器: {output_path}")
        
        # 修复：直接写入原始帧（避免颜色转换）
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)  # 直接写入原始帧
        
        cap.release()
        out.release()

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
    
    def save_batch(self, processed_videos: list, output_directory: str, prefix: str,
                  metadata="", start_index=1, digits=4, date_folder=True):
        # 创建基础目录
        output_directory = os.path.normpath(output_directory)
        os.makedirs(output_directory, exist_ok=True)
        
        # 按日期创建子目录
        if date_folder:
            date_str = datetime.now().strftime("%Y%m%d")
            output_directory = os.path.join(output_directory, date_str)
            os.makedirs(output_directory, exist_ok=True)
            self.batch_counter = 0  # 重置计数器
        
        # 设置起始索引
        current_index = max(start_index - 1, self.batch_counter)
        save_count = 0
        
        for idx, video in enumerate(processed_videos):
            try:
                # 生成序号
                index_str = str(current_index + idx + 1).zfill(digits)
                base_name = f"{prefix}_{index_str}"
                
                # 目标路径
                video_filename = f"{base_name}.{video['format']}"
                video_path = os.path.join(output_directory, video_filename)
                txt_path = os.path.join(output_directory, f"{base_name}_metadata.txt")
                
                # 复制视频文件
                if os.path.exists(video["output"]):
                    shutil.copy2(video["极output"], video_path)
                else:
                    raise FileNotFoundError(f"源文件不存在: {video['output']}")
                
                # 写入元数据
                if metadata:
                    self._write_metadata(txt_path, metadata, base_name, video)
                
                save_count += 1
                
            except Exception as e:
                continue
        
        # 更新计数器
        self.batch_counter += len(processed_videos)
        return ()
    
    def _write_metadata(self, path: str, metadata: str, base_name: str, video: dict):
        """生成元数据文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== 视频处理元数据 ===\n")
            f.write(f"文件名: {base_name}\n")
            f.write(f"格式: {video['format']}\n")
            f.write(f"源路径: {video['source']}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n自定义元数据:\n")
            f.write(metadata)

class TrueToSizeImageLoader:
    """真尺寸图像加载器 - 批量逐张输出原始尺寸图像"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "placeholder": "输入文件夹路径"}),
                "file_types": ("STRING", {"default": "jpg,png,jpeg,webp", "placeholder": "文件类型（逗号分隔）"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),  # 关键修复：step=1
                "reset_counter": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "loop_counter": ("INT", {"default": 0})  # 用于循环追踪
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "LIST")
    RETURN_NAMES = ("images", "masks", "current_index", "file_paths")
    OUTPUT_IS_LIST = (True, True, False, True)
    FUNCTION = "load_sequential_images"
    OUTPUT_NODE = True
    CATEGORY = "Zoey Tool/图像处理"
    
    def __init__(self):
        self.cached_file_list = []
        self.last_folder = ""
        self.last_types = ""

    def natural_sort_key(self, filename):
        """智能自然排序（正确排序数字序列）"""
        parts = re.split(r'(\d+)', os.path.basename(filename))
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    def update_file_list(self, folder_path, file_types):
        """更新文件列表（仅在必要时）"""
        if folder_path == self.last_folder and file_types == self.last_types and self.cached_file_list:
            return  # 使用缓存
        
        valid_exts = [ext.strip().lower() for ext in file_types.split(',')]
        self.cached_file_list = []
        
        # 获取所有匹配文件
        for ext in valid_exts:
            pattern = os.path.join(folder_path, f"*.{ext}")
            self.cached_file_list.extend(glob.glob(pattern, recursive=True))
        
        # 自然排序文件列表
        self.cached_file_list.sort(key=self.natural_sort_key)
        self.last_folder = folder_path
        self.last_types = file_types

    def load_sequential_images(self, folder_path, file_types, start_index, reset_counter, loop_counter=0):
        # 更新文件列表
        self.update_file_list(folder_path, file_types)
        total_images = len(self.cached_file_list)
        
        # 空文件夹处理
        if total_images == 0:
            return ((), (), 0, [])
        
        # 重置计数器（如果需要）
        if reset_counter:
            loop_counter = 0
        
        # 初始化输出列表
        image_list = []
        mask_list = []
        file_paths = []
        
        # 创建进度条
        progress = ProgressBar(total_images)
        
        for idx in range(loop_counter, min(loop_counter + total_images, total_images)):
            file_path = self.cached_file_list[idx]
            
            try:
                # 加载原始尺寸图像
                img = Image.open(file_path)
                
                # 分离alpha通道作为mask
                mask = None
                if img.mode == 'RGBA':
                    # 提取alpha通道作为mask
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    img = img.convert("RGB")
                
                # 转换为PyTorch张量（保持原始尺寸）
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W, C]
                
                image_list.append(img_tensor)
                file_paths.append(file_path)
                
                # 处理mask
                if mask is not None:
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
                    mask_list.append(mask_tensor)
                else:
                    # 创建全白mask
                    w, h = img.size
                    mask_list.append(torch.ones((1, h, w, 1), dtype=torch.float32))
                
                # 更新进度
                progress.update(1)
                
            except Exception as e:
                continue
        
        # 计算新的循环计数
        new_counter = loop_counter + len(image_list)
        
        # 完成时自动重置计数器
        if new_counter >= total_images:
            new_counter = 0
        
        return (image_list, mask_list, new_counter, file_paths)

class MultiFileBatchRenamer:
    """多文件批量处理器（支持后缀）"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "placeholder": "输入文件夹路径"}),
                "output_folder": ("STRING", {"default": "", "placeholder": "输出文件夹路径"}),
                "name_prefix": ("STRING", {"default": "FILE", "placeholder": "文件名前缀"}),
                "name_suffix": ("STRING", {"default": "", "placeholder": "文件名后缀（可选）"}),
                "start_index": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "sort_method": (["filename", "creation_date", "modification_date"], {"default": "filename"}),
                "name_mode": (["prefix_only", "prefix_timestamp"], {"default": "prefix_only"}),
                "file_types": ("STRING", {"default": "jpg,png,txt,pdf,doc", "placeholder": "支持的文件类型（逗号分隔）"})
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": False}),
                "padding_digits": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "content_check": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING", "INT", "LIST")
    RETURN_NAMES = ("output_folder", "file_count", "file_list")
    FUNCTION = "batch_rename"
    CATEGORY = "Zoey Tool/文件处理"
    OUTPUT_IS_LIST = (False, False, True)

    # 自然排序函数（修复排序问题的核心）
    def natural_sort_key(self, filename):
        parts = re.split(r'(\d+)', os.path.basename(filename))
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    def batch_rename(self, input_folder, output_folder, name_prefix, name_suffix, 
                    start_index, sort_method, name_mode, file_types, 
                    overwrite=False, padding_digits=4, content_check=False):
        # 路径验证
        if not os.path.exists(input_folder):
            raise ValueError(f"输入路径不存在: {input_folder}")
        if not output_folder:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_folder = os.path.join(os.path.dirname(input_folder), f"renamed_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        # 文件类型过滤
        valid_exts = [ext.strip().lower() for ext in file_types.split(',')]
        file_list = []
        for ext in valid_exts:
            file_list.extend(glob.glob(os.path.join(input_folder, f"*.{ext}")))
        if not file_list:
            return (output_folder, 0, [])
        
        # 文件名生成函数（含后缀处理）
        def generate_new_name(count, file_ext, conflict_count=0):
            count_str = str(count).zfill(padding_digits)
            suffix = f"_{name_suffix}" if name_suffix else ""
            
            if name_mode == "prefix_timestamp":
                timestamp = time.strftime("%Y%m%d%H%M%S")
                base_name = f"{name_prefix}_{timestamp}_{count_str}{suffix}"
            else:
                base_name = f"{name_prefix}{count_str}{suffix}"
            
            conflict_mark = f"_{conflict_count}" if conflict_count > 0 else ""
            return f"{base_name}{conflict_mark}{file_ext}"

        # 修复：使用自然排序确保数字顺序正确
        if sort_method == "creation_date":
            file_list.sort(key=lambda x: os.path.getctime(x))
        elif sort_method == "modification_date":
            file_list.sort(key=lambda x: os.path.getmtime(x))
        else:
            file_list.sort(key=self.natural_sort_key)  # 关键修复点
        
        # 文件处理主循环
        count = start_index
        processed_files = []
        
        for file_path in file_list:
            try:
                filename = os.path.basename(file_path)
                file_ext = os.path.splitext(filename)[1].lower()
                
                new_name = generate_new_name(count, file_ext)
                output_path = os.path.join(output_folder, new_name)
                
                # 冲突处理
                conflict_count = 0
                while os.path.exists(output_path) and not overwrite:
                    conflict_count += 1
                    new_name = generate_new_name(count, file_ext, conflict_count)
                    output_path = os.path.join(output_folder, new_name)
                
                # 特殊文件处理
                if file_ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
                    img = Image.open(file_path)
                    img.save(output_path)
                elif content_check and file_ext in ['.txt', '.csv', '.md']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                    if re.search(r'[^\x00-\x7F]', content):
                        new_name = f"UTF8_{new_name}"
                        output_path = os.path.join(output_folder, new_name)
                    shutil.copy2(file_path, output_path)
                else:
                    shutil.copy2(file_path, output_path)
                
                processed_files.append(output_path)
                count += 1
            except Exception as e:
                continue
        
        return (output_folder, len(processed_files), processed_files)

class PureTranslator:
    """纯净翻译器"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "source_lang": (["auto", "中文", "英文", "日语", "韩语"], {"default": "auto"}),
                "target_lang": (["英文", "中文", "日语", "韩语"], {"default": "英文"}),
                "engine": (["内置模型", "百度API", "谷歌API(免费)"], {"default": "内置模型"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "label": "API密钥(仅百度API需要)"}),
                "max_length": ("INT", {"default": 120, "min": 50, "max": 300}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("纯净翻译文本", "处理耗时(秒)")
    FUNCTION = "pure_translate"
    CATEGORY = "Zoey Tool/文本处理"
    OUTPUT_NODE = True

    def __init__(self):
        self.translator = None
        self.last_source = ""
        self.last_target = ""

    def load_nmt_model(self, source, target):
        """安全加载内置翻译模型"""
        model_map = {
            ("auto", "英文"): "Helsinki-NLP/opus-mt-mul-en",
            ("中文", "英文"): "Helsinki-NLP/opus-mt-zh-en",
            ("英文", "中文"): "Helsinki-NLP/opus-mt-en-zh",
            ("日语", "英文"): "Helsinki-NLP/opus-mt-ja-en",
            ("英文", "日语"): "Helsinki-NLP/opus-mt-en-ja",
            ("韩语", "英文"): "Helsinki-NLP/opus-mt-ko-en",
            ("英文", "韩语"): "Helsinki-NLP/opus-mt-en-ko",
        }
        
        # 确定最适合的模型
        if (source, target) in model_map:
            model_name = model_map[(source, target)]
        elif source == "auto" and target != "英文":
            model_name = model_map[("auto", "英文")]  # 自动检测转英文
        else:
            model_name = "Helsinki-NLP/opus-mt-en-zh"  # 默认回退
            
        try:
            self.translator = pipeline(
                "translation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.last_source = source
            self.last_target = target
            return True
        except Exception as e:
            return False

    def baidu_translate(self, text, api_key, target_lang):
        """可靠的百度翻译API实现"""
        if not api_key or ':' not in api_key:
            return "❌❌❌❌ 无效API密钥格式，应为app_id:secret_key"
            
        app_id, secret_key = api_key.split(':', 1)
        url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        salt = str(int(time.time() * 1000))
        sign_str = app_id + text + salt + secret_key
        sign = hashlib.md5(sign_str.encode()).hexdigest()
        
        lang_map = {"中文": "zh", "英文": "en", "日语": "jp", "韩语": "kor"}
        to_lang = lang_map.get(target_lang, "en")
        
        params = {
            "q": text,
            "from": "auto",
            "to": to_lang,
            "appid": app_id,
            "salt": salt,
            "sign": sign
        }
        
        try:
            response = requests.post(url, data=params, timeout=8)
            response.raise_for_status()
            result = response.json()
            return "".join([res["dst"] for res in result["trans_result"]])
        except Exception as e:
            return f"❌❌❌❌ 翻译失败: {str(e)}"

    def google_translate(self, text, target_lang):
        """免费的谷歌翻译API实现"""
        lang_codes = {"英文": "en", "中文": "zh-CN", "日语": "ja", "韩语": "ko"}
        lang_code = lang_codes.get(target_lang, "en")
        
        url = f"https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": lang_code,
            "dt": "t",
            "q": text
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return "".join([s[0] for s in data[0]])
            else:
                return "❌❌❌❌ 谷歌翻译API错误"
        except:
            return "❌❌❌❌ 无法连接谷歌服务"

    def pure_translate(self, text, source_lang, target_lang, engine, api_key="", max_length=120):
        # 统一使用start_time计算耗时
        start_time = time.time()  # 修复变量名
        
        if not text.strip():
            return ("", 0.0)
        
        result = ""
        
        try:
            # 内置模型处理
            if engine == "内置模型":
                if self.load_nmt_model(source_lang, target_lang):
                    # 限制过长的输入文本
                    safe_text = text[:200] if len(text) > 200 else text
                    result = self.translator(safe_text)[0]['translation_text']
                else:
                    result = "❌❌❌❌ 无法加载内置翻译模型"
            # 百度API处理
            elif engine == "百度API":
                result = self.baidu_translate(text, api_key, target_lang)
            # 谷歌API处理
            elif engine == "谷歌API(免费)":
                result = self.google_translate(text, target_lang)
            
            # 处理耗时计算
            proc_time = time.time() - start_time
            
            # 确保结果为字符串并清理
            result = str(result).strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            return (result, round(proc_time, 3))
            
        except Exception as e:
            return (f"❌❌❌❌ 处理出错: {str(e)}", 0.0)

class GuaranteedBatchSaver:
    """
    确保批量完整保存器 - 解决15张只保存10张的问题
    支持自定义起始序号和TXT开关控制
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "save_path": ("STRING", {"default": "D:/11aa打标"}),
                "prefix": ("STRING", {"default": "batch"}),
                "separator": ("STRING", {"default": "_"}),
                "digits": ("INT", {"default": 5, "min": 1, "max": 8}),
                "text_suffix": ("STRING", {"default": "txt"}),
                "start_index": ("INT", {"default": 1, "min": 0, "step": 1}),
                "save_txt": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),  # 明确开关标签
            },
            "optional": {
                "txte": ("STRING", {"multiline": True, "default": ""}),
                "texts": ("STRING", {"multiline": True, "default": ""}),
                "batch_name": ("STRING", {"default": "train", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "guaranteed_batch_save"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True
    
    def __init__(self):
        self.last_save_time = 0
        self.file_count_map = {}
    
    def get_safe_start_index(self, save_path, prefix, separator, digits):
        """安全获取下一个起始序号（兼容手动设置）"""
        if not os.path.exists(save_path):
            return 0
        
        all_files = os.listdir(save_path)
        pattern = re.compile(rf"^{re.escape(prefix)}{re.escape(separator)}(\d{{{digits}}})")
        max_index = -1
        for filename in all_files:
            match = pattern.match(filename)
            if match:
                try:
                    file_index = int(match.group(1))
                    if file_index > max_index:
                        max_index = file_index
                except ValueError:
                    continue
        return max_index + 1 if max_index >= 0 else 0

    def format_index(self, index, digits):
        """根据digits参数格式化序号"""
        return str(index).zfill(digits) if digits > 1 else str(index)
    
    def guaranteed_batch_save(self, images, save_path, prefix, separator, 
                             digits, text_suffix, start_index=1,
                             txte="", texts="", batch_name="train",
                             save_txt=True):  # 确保参数正确传递
        
        os.makedirs(save_path, exist_ok=True)
        batch_size = images.shape[0] if images.dim() == 4 else 1
        current_time = time.time()
        
        # 双重机制 + 自定义起始序号逻辑
        if batch_size > 10 or (current_time - self.last_save_time > 5):
            auto_index = self.get_safe_start_index(save_path, prefix, separator, digits)
            # 优先使用用户设置的起始序号（若大于自动计算的序号）
            start_index = max(auto_index, start_index)
            self.file_count_map[save_path] = start_index
        else:
            start_index = self.file_count_map.get(save_path, start_index)  # 延续上次序号
        
        full_prefix = f"{batch_name}{separator}{prefix}" if batch_name else prefix
        
        saved_count = 0
        for i in range(batch_size):
            current_index = start_index + i
            formatted_index = self.format_index(current_index, digits)
            base_filename = f"{full_prefix}{separator}{formatted_index}"
            
            # 保存图像
            img_data = images[i] if images.dim() == 4 else images
            img_data = img_data.cpu().numpy() * 255.0
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_data, "RGBA" if img_data.shape[-1] == 4 else "RGB")
            img_path = os.path.join(save_path, f"{base_filename}.png")
            img_pil.save(img_path)
            
            # 关键修复：根据save_txt开关控制TXT输出
            if save_txt:
                # 保存主文本文件
                txt_path = os.path.join(save_path, f"{base_filename}.{text_suffix}")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(txte.strip())
                
                # 保存备用文本文件（仅当有内容时）
                if texts.strip():
                    alt_path = os.path.join(save_path, f"{base_filename}_alt.{text_suffix}")
                    with open(alt_path, "w", encoding="utf-8") as f:
                        f.write(texts.strip())
            
            saved_count += 1
        
        # 更新状态
        self.last_save_time = current_time
        self.file_count_map[save_path] = start_index + batch_size
        return (images,)
# ======================
# 节点注册
# ======================
NODE_CLASS_MAPPINGS = {
    "WanPromptGenerator": WanPromptGenerator,
    "BatchVideoLoader": BatchVideoLoader,
    "VideoBatchProcessor": VideoBatchProcessor,
    "EnhancedVideoSaver": EnhancedVideoSaver,
    "TrueToSizeImageLoader": TrueToSizeImageLoader,
    "MultiFileBatchRenamer": MultiFileBatchRenamer,
    "PureTranslator": PureTranslator,
    "GuaranteedBatchSaver": GuaranteedBatchSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanPromptGenerator": "Zoey - Wan2.2提示词生成器",
    "BatchVideoLoader": "Zoey - 批量视频加载器",
    "VideoBatchProcessor": "Zoey - 视频批处理器",
    "EnhancedVideoSaver": "Zoey - 智能视频存储器",
    "TrueToSizeImageLoader": "Zoey - 真尺寸图像加载器",
    "MultiFileBatchRenamer": "Zoey - 多文件批量处理器",
    "PureTranslator": "Zoey - 纯净翻译器",
    "GuaranteedBatchSaver": "Zoey - 智能图像存储器"
}
