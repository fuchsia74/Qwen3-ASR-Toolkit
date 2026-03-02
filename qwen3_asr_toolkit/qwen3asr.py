import os
import time
import random
import re
import base64
import mimetypes
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
import requests
import dashscope


MAX_API_RETRY = 10
API_RETRY_SLEEP = (1, 2)


language_code_mapping = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish"
}


class QwenASR:
    def __init__(
        self,
        provider: str,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 300,
        temperature: Optional[float] = None,
        max_retries: int = MAX_API_RETRY,
    ):
        self.provider = provider
        self.api_url = api_url
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_retries = max_retries

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def _normalize_content(self, content):
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _redact_base64(self, text: str) -> str:
        if not text:
            return text
        return re.sub(r"[A-Za-z0-9+/=]{80,}", "<base64-redacted>", text)

    def _summarize_error(self, error: Exception) -> str:
        message = f"{error.__class__.__name__}: {error}"
        message = self._redact_base64(message)
        if len(message) > 300:
            message = message[:300] + "...(truncated)"
        return message

    def _display_audio_ref(self, audio_ref: str) -> str:
        if not audio_ref:
            return audio_ref
        if audio_ref.startswith("data:"):
            return "<data-url-audio>"
        return audio_ref

    def _parse_asr_output(self, content):
        try:
            from qwen_asr import parse_asr_output  # type: ignore
            return parse_asr_output(content)
        except Exception:
            pass

        content = self._normalize_content(content).strip()
        lang_match = re.search(r"(?i)language\s*[:=]\s*([A-Za-z-]+)", content)
        text_match = re.search(r"(?i)text\s*[:=]\s*(.+)", content, re.S)
        lang_code = lang_match.group(1).lower() if lang_match else None
        language = language_code_mapping.get(lang_code, "Unknown")
        text = text_match.group(1).strip() if text_match else content
        return language, text

    def _to_data_url(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _asr_local(self, wav_url: str, context: str = ""):
        if not wav_url.startswith("http"):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"
            file_path = wav_url
            file_size = os.path.getsize(file_path)

            # file size > 10M
            if file_size > 10 * 1024 * 1024:
                # convert to mp3
                mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                audio = AudioSegment.from_file(file_path)
                audio.export(mp3_path, format="mp3")
                wav_url = mp3_path

            wav_url = self._to_data_url(wav_url)

        # Submit the ASR task
        response = None
        display_ref = self._display_audio_ref(wav_url)
        for _ in range(self.max_retries):
            try:
                messages = []
                if context:
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": context}],
                    })
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "audio_url",
                        "audio_url": {"url": wav_url},
                    }],
                })
                payload = {"messages": messages}
                if self.model:
                    payload["model"] = self.model
                if self.temperature is not None:
                    payload["temperature"] = self.temperature

                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                response_json = response.json()
                output = response_json["choices"][0]["message"]["content"]
                language, recog_text = self._parse_asr_output(output)
                return language, self.post_text_process(recog_text)
            except Exception as e:
                try:
                    status_code = getattr(response, "status_code", "unknown")
                    reason = getattr(response, "reason", "")
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {display_ref}\nHTTP {status_code} {reason}\n{error_summary}")
                except Exception:
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {display_ref}\n{error_summary}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{wav_url} task failed!\n{response}")

    def _asr_dashscope(self, wav_url: str, context: str = ""):
        if not self.model:
            raise ValueError("DashScope requires a model name.")
        if not wav_url.startswith("http"):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"
            file_path = wav_url
            file_size = os.path.getsize(file_path)

            # file size > 10M
            if file_size > 10 * 1024 * 1024:
                # convert to mp3
                mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                audio = AudioSegment.from_file(file_path)
                audio.export(mp3_path, format="mp3")
                wav_url = mp3_path

            wav_url = f"file://{wav_url}"

        response = None
        for _ in range(self.max_retries):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"text": context},
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"audio": wav_url},
                        ]
                    }
                ]
                response = dashscope.MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    result_format="message",
                    asr_options={
                        "enable_lid": True,
                        "enable_itn": False
                    }
                )

                if response.status_code != 200:
                    raise Exception(f"http status_code: {response.status_code} {response}")
                output = response['output']['choices'][0]

                recog_text = None
                if len(output["message"]["content"]):
                    recog_text = output["message"]["content"][0]["text"]
                if recog_text is None:
                    recog_text = ""

                lang_code = None
                if "annotations" in output["message"]:
                    lang_code = output["message"]["annotations"][0]["language"]
                language = language_code_mapping.get(lang_code, "Not Supported")

                return language, self.post_text_process(recog_text)
            except Exception as e:
                try:
                    print(f"Retry {_ + 1}...  {wav_url}\n{response}")
                except Exception:
                    error_summary = self._summarize_error(e)
                    print(f"Retry {_ + 1}...  {error_summary}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{wav_url} task failed!\n{response}")

    def asr(self, wav_url: str, context: str = ""):
        if self.provider == "local":
            return self._asr_local(wav_url, context)
        if self.provider == "dashscope":
            return self._asr_dashscope(wav_url, context)
        raise ValueError(f"Unknown provider: {self.provider}")


if __name__ == "__main__":
    qwen_asr = QwenASR(provider="local", api_url="http://localhost:8000/v1/chat/completions", model=None)
    asr_text = qwen_asr.asr(wav_url="/path/to/your/wav_file.wav")
    print(asr_text)
