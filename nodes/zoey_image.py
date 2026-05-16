# -*- coding: utf-8 -*-
"""
Zoey Image Nodes - 图像处理专用节点
包含：真尺寸加载、智能存储
"""

import os
import torch
import numpy as np
import glob
import re
import time
from PIL import Image
from comfy.utils import ProgressBar

# ====================== 节点 4: 真尺寸图像加载器 ======================
class TrueToSizeImageLoader:
    """真尺寸图像加载器 - 批量逐张输出原始尺寸图像"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "placeholder": "输入文件夹路径"}),
                "file_types": ("STRING", {"default": "jpg,png,jpeg,webp", "placeholder": "文件类型（逗号分隔）"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "reset_counter": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "loop_counter": ("INT", {"default": 0})
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
        parts = re.split(r'(\d+)', os.path.basename(filename))
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    def update_file_list(self, folder_path, file_types):
        if folder_path == self.last_folder and file_types == self.last_types and self.cached_file_list:
            return
        valid_exts = [ext.strip().lower() for ext in file_types.split(',')]
        self.cached_file_list = []
        for ext in valid_exts:
            pattern = os.path.join(folder_path, f"*.{ext}")
            self.cached_file_list.extend(glob.glob(pattern, recursive=True))
        self.cached_file_list.sort(key=self.natural_sort_key)
        self.last_folder = folder_path
        self.last_types = file_types

    def load_sequential_images(self, folder_path, file_types, start_index, reset_counter, loop_counter=0):
        self.update_file_list(folder_path, file_types)
        total_images = len(self.cached_file_list)
        
        if total_images == 0:
            return ((), (), 0, [])
            
        if reset_counter:
            loop_counter = 0
            
        image_list = []
        mask_list = []
        file_paths = []
        progress = ProgressBar(total_images)
        
        for idx in range(loop_counter, min(loop_counter + total_images, total_images)):
            file_path = self.cached_file_list[idx]
            try:
                img = Image.open(file_path)
                mask = None
                if img.mode == 'RGBA':
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                image_list.append(img_tensor)
                file_paths.append(file_path)
                
                if mask is not None:
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
                    mask_list.append(mask_tensor)
                else:
                    w, h = img.size
                    mask_list.append(torch.ones((1, h, w, 1), dtype=torch.float32))
                progress.update(1)
            except Exception as e:
                continue
                
        new_counter = loop_counter + len(image_list)
        if new_counter >= total_images:
            new_counter = 0
        return (image_list, mask_list, new_counter, file_paths)

# ====================== 节点 8: 智能图像存储器 (已更新) ======================
class GuaranteedBatchSaver:
    """确保批量完整保存器 - 增加覆盖模式开关"""
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
                "save_txt": ("BOOLEAN", {"default": True}),
                # 👇 新增：覆盖模式开关
                "overwrite_mode": ("BOOLEAN", {"default": False, "label_on": "开启覆盖", "label_off": "顺序保存"})
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

    def format_index(self, index, digits):
        return str(index).zfill(digits) if digits > 1 else str(index)

    def guaranteed_batch_save(self, images, save_path, prefix, separator, digits, text_suffix, start_index=1, txte="", texts="", batch_name="train", save_txt=True, overwrite_mode=False):
        os.makedirs(save_path, exist_ok=True)
        batch_size = images.shape[0] if images.dim() == 4 else 1
        
        # 👇 核心逻辑：如果开启覆盖模式，强制从 start_index 开始；否则按计数器递增
        if overwrite_mode:
            current_start = start_index
            print(f"[Zoey Saver] 覆盖模式开启，文件将从索引 {start_index} 开始写入...")
        else:
            # 顺序保存逻辑（原逻辑）
            if save_path not in self.file_count_map:
                self.file_count_map[save_path] = start_index
            current_start = self.file_count_map[save_path]
        
        full_prefix = f"{batch_name}{separator}{prefix}" if batch_name else prefix
        saved_count = 0
        
        for i in range(batch_size):
            current_index = current_start + i
            formatted_index = self.format_index(current_index, digits)
            base_filename = f"{full_prefix}{separator}{formatted_index}"
            
            # 保存图像
            img_data = images[i] if images.dim() == 4 else images
            img_data = img_data.cpu().numpy() * 255.0
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_data, "RGBA" if img_data.shape[-1] == 4 else "RGB")
            img_path = os.path.join(save_path, f"{base_filename}.png")
            img_pil.save(img_path)
            
            # 条件保存文本文件
            if save_txt:
                txte_path = os.path.join(save_path, f"{base_filename}{separator}{text_suffix}.txt")
                with open(txte_path, "w", encoding="utf-8") as f:
                    f.write(txte)
                if texts.strip():
                    texts_path = os.path.join(save_path, f"{base_filename}{separator}{text_suffix}_alt.txt")
                    with open(texts_path, "w", encoding="utf-8") as f:
                        f.write(texts)
            saved_count += 1
            
        # 👇 只有在非覆盖模式下才更新计数器，覆盖模式下计数器保持不变
        if not overwrite_mode:
            self.file_count_map[save_path] = current_start + batch_size
            
        self.last_save_time = time.time()
        return (images,)

# ====================== 注册 ======================
NODE_CLASS_MAPPINGS = {
    "TrueToSizeImageLoader": TrueToSizeImageLoader,
    "GuaranteedBatchSaver": GuaranteedBatchSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrueToSizeImageLoader": "Zoey - 真尺寸图像加载器",
    "GuaranteedBatchSaver": "Zoey - 智能图像存储器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']