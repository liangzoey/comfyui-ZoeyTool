import math
import torch
import comfy.utils
import node_helpers

class ZOEYTextEncodeQwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {"vae": ("VAE", ),
                         "image1": ("IMAGE", ),
                         "image2": ("IMAGE", ),
                         "image3": ("IMAGE", ),
                         "image4": ("IMAGE", ),
                         "image5": ("IMAGE", ),
                         "image6": ("IMAGE", ),
                         }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    
    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, image5=None, image6=None):
        ref_latents = []
        images = [image1, image2, image3, image4, image5, image6]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""
        
        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
                
                image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
        
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning, )

NODE_CLASS_MAPPINGS = {
    "ZOEYTextEncodeQwenImageEditPlus": ZOEYTextEncodeQwenImageEditPlus,
}