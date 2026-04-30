import os
import torch
import numpy as np
import folder_paths
from PIL import Image

# ComfyUI 会自动将此目录下的 JS/CSS 映射到 /extensions/zoey_vr/
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

class VR360PreviewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "zoey/VR"

    def preview(self, image):
        output_dir = folder_paths.get_output_directory()
        filename_prefix = "vr_preview"
        # image.shape: (B, H, W, C) -> 宽, 高
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, image.shape[2], image.shape[1]
        )
        
        results = []
        for batch_number, i in enumerate(image):
            img = Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8))
            file = f"{filename}_{batch_number:05d}.png"
            img.save(os.path.join(full_output_folder, file), quality=95)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            
        # 核心：返回 ui.images，ComfyUI 前端会自动填充 node.imgs
        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {"VR360PreviewNode": VR360PreviewNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VR360PreviewNode": "zoey VR 360° 嵌入式预览"}
