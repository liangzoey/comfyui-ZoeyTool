# zoey_hunyuan_translator.py
# 独立 ComfyUI 节点：腾讯混元 HY-MT1.5 本地化翻译器
# 模型自动下载并缓存至: F:\ComfyUINeo\ComfyUINeo\models\llm

import os
import torch
import time
from huggingface_hub import snapshot_download

# 尝试导入 ComfyUI 所需模块（避免在非 Comfy 环境报错）
try:
    import folder_paths
except ImportError:
    folder_paths = None

class HunyuanTranslatorNode:
    """腾讯混元 HY-MT1.5 翻译节点 - 自动下载 + 本地加载"""

    # 固定模型存储路径（根据你的需求）
    LOCAL_MODEL_ROOT = r"F:\ComfyUINeo\ComfyUINeo\models\llm"

    # 类级缓存，避免重复加载
    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
                "源语言": (["自动检测", "中文", "英文", "日语", "韩语", "法语", "西班牙语", "俄语", "阿拉伯语", "德语", "葡萄牙语", "意大利语", "泰语", "越南语", "印尼语", "繁体中文", "粤语", "藏语", "维吾尔语", "蒙古语", "哈萨克语"],),
                "目标语言": (["中文", "英文", "日语", "韩语", "法语", "西班牙语", "俄语", "阿拉伯语", "德语", "葡萄牙语", "意大利语", "泰语", "越南语", "印尼语", "繁体中文", "粤语", "藏语", "维吾尔语", "蒙古语", "哈萨克语"], {"default": "英文"}),
                "模型版本": (["HY-MT1.5-1.8B", "HY-MT1.5-7B"], {"default": "HY-MT1.5-1.8B"}),
                "设备": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "术语干预": ("STRING", {"multiline": True, "default": "", "placeholder": "例: AI -> 人工智能\nGPU -> 图形处理器"}),
                "上下文": ("STRING", {"multiline": True, "default": "", "placeholder": "提供前文上下文（用于上下文翻译）"}),
                "保留格式标签": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "最大新令牌数": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("翻译结果", "耗时(秒)")
    FUNCTION = "translate"
    CATEGORY = "Zoey Tool/文本处理"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    def _get_lang_code(self, lang_name: str) -> str:
        lang_map = {
            "中文": "zh", "英文": "en", "日语": "ja", "韩语": "ko", "法语": "fr", "西班牙语": "es",
            "俄语": "ru", "阿拉伯语": "ar", "德语": "de", "葡萄牙语": "pt", "意大利语": "it",
            "泰语": "th", "越南语": "vi", "印尼语": "id", "繁体中文": "zh-Hant", "粤语": "yue",
            "藏语": "bo", "维吾尔语": "ug", "蒙古语": "mn", "哈萨克语": "kk", "自动检测": "auto"
        }
        return lang_map.get(lang_name, "en")

    def _get_local_model_path(self, model_version: str):
        local_dir = os.path.join(self.LOCAL_MODEL_ROOT, model_version)
        config_path = os.path.join(local_dir, "config.json")
        if os.path.exists(config_path):
            print(f"[HunyuanTranslator] 使用本地模型: {local_dir}")
            return local_dir

        print(f"[HunyuanTranslator] 本地模型未找到，正在下载 '{model_version}' 到 {local_dir} ...")
        hf_model_map = {
            "HY-MT1.5-1.8B": "tencent/HY-MT1.5-1.8B",
            "HY-MT1.5-7B": "tencent/HY-MT1.5-7B"
        }
        hf_repo_id = hf_model_map[model_version]

        os.makedirs(self.LOCAL_MODEL_ROOT, exist_ok=True)
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                # endpoint="https://hf-mirror.com"  # 如需国内镜像，取消注释
            )
            print(f"[HunyuanTranslator] 模型下载完成: {local_dir}")
            return local_dir
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {e}. 请检查网络或手动放置模型到 {local_dir}")

    def _load_model(self, model_path: str, device: str):
        cache_key = f"{model_path}_{device}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[HunyuanTranslator] 正在加载模型: {model_path}")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        device_map = "auto" if device == "auto" else device
        torch_dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        model.eval()
        self._model_cache[cache_key] = (model, tokenizer)
        print(f"[HunyuanTranslator] 模型加载完成，耗时 {time.time() - start:.2f} 秒")
        return model, tokenizer

    def _build_prompt(self, text: str, src_lang: str, tgt_lang: str, terminology: str = "", context: str = "", preserve_format: bool = False):
        lang_name_map = {
            "zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어", "fr": "Français", "es": "Español",
            "ru": "Русский", "ar": "العربية", "de": "Deutsch", "pt": "Português", "it": "Italiano", "th": "ไทย",
            "vi": "Tiếng Việt", "id": "Bahasa Indonesia", "zh-Hant": "繁體中文", "yue": "粵語",
            "bo": "བོད་སྐད", "ug": "ئۇيغۇرچە", "mn": "Монгол", "kk": "Қазақ"
        }
        tgt_lang_name = lang_name_map.get(tgt_lang, tgt_lang)

        if terminology.strip():
            term_lines = []
            for line in terminology.strip().split('\n'):
                if '->' in line:
                    parts = line.split('->', 1)
                    if len(parts) == 2:
                        s, t = parts
                        term_lines.append(f"{s.strip()} 翻译成 {t.strip()}")
            prompt = f"参考下面的翻译：\n{'\n'.join(term_lines)}\n\n将以下文本翻译为{tgt_lang_name}，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
        elif context.strip():
            prompt = f"{context}\n参考上面的信息，把下面的文本翻译成{tgt_lang_name}，注意不需要翻译上文，也不要额外解释：\n{text}"
        elif preserve_format:
            prompt = f"将以下<source></source>之间的文本翻译为{tgt_lang_name}，注意只需要输出翻译后的结果，不要额外解释，原文中的<sn></sn>标签表示标签内文本包含格式信息，需要在译文中相应的位置尽量保留该标签。输出格式为：<target>str</target>\n\n<source>{text}</source>"
        else:
            if src_lang == "zh":
                prompt = f"将以下文本翻译为{tgt_lang_name}，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
            else:
                prompt = f"Translate the following segment into {tgt_lang_name}, without additional explanation.\n\n{text}"
        return prompt

    def translate(self, 文本, 源语言, 目标语言, 模型版本, 设备, 术语干预="", 上下文="", 保留格式标签=False, 最大新令牌数=512):
        if not 文本.strip():
            return ("", 0.0)

        start_time = time.time()
        src_code = self._get_lang_code(源语言)
        tgt_code = self._get_lang_code(目标语言)

        model_path = self._get_local_model_path(模型版本)
        model, tokenizer = self._load_model(model_path, 设备)

        prompt = self._build_prompt(文本, src_code, tgt_code, 术语干预, 上下文, 保留格式标签)

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=最大新令牌数,
                do_sample=True,
                top_k=20,
                top_p=0.6,
                temperature=0.7,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成内容（去除 prompt 和系统消息）
        if "<|im_start|>assistant" in output_text:
            output_text = output_text.split("<|im_start|>assistant")[-1].strip()
        else:
            # 启发式截断输入部分
            output_text = output_text[len(prompt):].strip()

        # 处理格式保留模式
        if 保留格式标签 and output_text.startswith("<target>") and "</target>" in output_text:
            try:
                output_text = output_text.split("<target>", 1)[1].split("</target>", 1)[0]
            except:
                pass  # 若解析失败，保留原输出

        result = output_text.strip()
        elapsed = round(time.time() - start_time, 3)
        return (result, elapsed)


# ======================
# ComfyUI 节点注册
# ======================

NODE_CLASS_MAPPINGS = {
    "HunyuanTranslatorNode": HunyuanTranslatorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanTranslatorNode": "Zoey - 混元翻译器 (HY-MT1.5)"
}