import os
import time
import hashlib
import torch
import requests
from transformers import pipeline


class PureTranslator:
    """纯净翻译器 - 支持内置模型 / 百度API / 谷歌API"""
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
        model_map = {
            ("auto", "英文"): "Helsinki-NLP/opus-mt-mul-en",
            ("中文", "英文"): "Helsinki-NLP/opus-mt-zh-en",
            ("英文", "中文"): "Helsinki-NLP/opus-mt-en-zh",
            ("日语", "英文"): "Helsinki-NLP/opus-mt-ja-en",
            ("英文", "日语"): "Helsinki-NLP/opus-mt-en-ja",
            ("韩语", "英文"): "Helsinki-NLP/opus-mt-ko-en",
            ("英文", "韩语"): "Helsinki-NLP/opus-mt-en-ko",
        }

        if (source, target) in model_map:
            model_name = model_map[(source, target)]
        elif source == "auto" and target != "英文":
            model_name = model_map[("auto", "英文")]
        else:
            model_name = "Helsinki-NLP/opus-mt-en-zh"

        try:
            self.translator = pipeline(
                "translation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.last_source = source
            self.last_target = target
            return True
        except Exception:
            return False

    def baidu_translate(self, text, api_key, target_lang):
        if not api_key or ':' not in api_key:
            return "API密钥格式错误，应为app_id:secret_key"

        app_id, secret_key = api_key.split(':', 1)
        url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        salt = str(int(time.time() * 1000))
        sign_str = app_id + text + salt + secret_key
        sign = hashlib.md5(sign_str.encode()).hexdigest()

        lang_map = {"中文": "zh", "英文": "en", "日语": "jp", "韩语": "kor"}
        to_lang = lang_map.get(target_lang, "en")

        params = {
            "q": text, "from": "auto", "to": to_lang,
            "appid": app_id, "salt": salt, "sign": sign
        }

        try:
            response = requests.post(url, data=params, timeout=8)
            response.raise_for_status()
            result = response.json()
            return "".join([res["dst"] for res in result["trans_result"]])
        except Exception as e:
            return f"翻译失败: {str(e)}"

    def google_translate(self, text, target_lang):
        lang_codes = {"英文": "en", "中文": "zh-CN", "日语": "ja", "韩语": "ko"}
        lang_code = lang_codes.get(target_lang, "en")

        params = {
            "client": "gtx", "sl": "auto", "tl": lang_code,
            "dt": "t", "q": text
        }

        try:
            response = requests.get(
                "https://translate.googleapis.com/translate_a/single",
                params=params, timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return "".join([s[0] for s in data[0]])
            return "谷歌翻译API错误"
        except Exception:
            return "无法连接谷歌服务"

    def pure_translate(self, text, source_lang, target_lang, engine, api_key="", max_length=120):
        start_time = time.time()

        if not text.strip():
            return ("", 0.0)

        result = ""
        try:
            if engine == "内置模型":
                if self.load_nmt_model(source_lang, target_lang):
                    safe_text = text[:200] if len(text) > 200 else text
                    result = self.translator(safe_text)[0]['translation_text']
                else:
                    result = "无法加载内置翻译模型"
            elif engine == "百度API":
                result = self.baidu_translate(text, api_key, target_lang)
            elif engine == "谷歌API(免费)":
                result = self.google_translate(text, target_lang)

            proc_time = time.time() - start_time
            result = str(result).strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]

            return (result, round(proc_time, 3))

        except Exception as e:
            return (f"处理出错: {str(e)}", 0.0)


NODE_CLASS_MAPPINGS = {
    "PureTranslator": PureTranslator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PureTranslator": "Zoey - 纯净翻译器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
