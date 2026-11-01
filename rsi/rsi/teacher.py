"""Client for the frozen teacher pi_T (any OpenAI-compatible chat endpoint, e.g. vLLM or SGLang serving open weights).

The teacher's weights and decoding settings never change during the loop; only the harness H does.  A harness calls
`Teacher(cfg_teacher).chat(messages, images=...)` to get the next tool call.
"""
import base64
import json
import os
import urllib.request


class Teacher:
    def __init__(self, t=None):
        t = t or {}
        self.base_url = (t.get("base_url") or os.environ.get("TEACHER_BASE_URL", "")).rstrip("/")
        self.model = t.get("model") or os.environ.get("TEACHER_MODEL", "")
        self.temperature = float(t.get("temperature", os.environ.get("TEACHER_TEMPERATURE", 0)))
        self.max_tokens = int(t.get("max_tokens", 4096))
        self.extra = t.get("extra_body") or {}
        self.key = os.environ.get(t.get("api_key_env", "TEACHER_API_KEY"), "EMPTY")

    def chat(self, messages, images=(), timeout=600):
        """messages: OpenAI-style list; images: JPEG/PNG bytes attached to the last user message."""
        if images:
            last = messages[-1]
            parts = [{"type": "text", "text": last["content"]}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
                for b in images]
            messages = messages[:-1] + [{"role": last["role"], "content": parts}]
        body = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, **self.extra)
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
