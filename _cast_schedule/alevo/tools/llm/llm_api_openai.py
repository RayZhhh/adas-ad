from __future__ import annotations

import openai
from typing import Any

from alevo.base import Sampler


class OpenAIAPI(Sampler):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=30, sys='', **kwargs):
        super().__init__()
        self._timeout = timeout
        self._model = model
        self._sys = sys
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        prompt_ = []
        if self._sys != '':
            prompt_.append({'role': 'system', 'content': self._sys})
        prompt_.append({'role': 'user', 'content': prompt.strip()})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=prompt_,
            stream=False,
            timeout=self._timeout
        )
        return response.choices[0].message.content
