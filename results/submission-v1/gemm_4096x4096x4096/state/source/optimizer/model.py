"""Explicit model-file configuration; never reads personal harness settings."""
import json
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelConfig:
    protocol: str
    base_url: str
    model: str
    api_key_env: str
    max_tokens_field: str = 'max_completion_tokens'
    trust_env: bool = False
    thinking: str | None = None

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            data = json.load(stream)
        fields = ('protocol', 'base_url', 'model', 'api_key_env')
        if any(not isinstance(data.get(k), str) or not data[k].strip() for k in fields):
            raise ValueError('Model config requires nonempty strings: ' + ', '.join(fields))
        config = cls(**{k: data[k] for k in fields},
                     max_tokens_field=data.get('max_tokens_field', 'max_completion_tokens'),
                     trust_env=data.get('trust_env', False), thinking=data.get('thinking'))
        if config.thinking not in {None, 'enabled', 'disabled'}:
            raise ValueError('thinking must be enabled or disabled')
        if config.thinking is not None and config.protocol != 'openai_chat_completions':
            raise ValueError('thinking extension is supported only for Chat Completions')
        if config.max_tokens_field not in {'max_tokens', 'max_completion_tokens'}:
            raise ValueError('Unsupported max_tokens_field')
        if not isinstance(config.trust_env, bool):
            raise ValueError('trust_env must be boolean')
        if config.protocol not in {'openai_responses', 'openai_chat_completions', 'anthropic_messages'}:
            raise ValueError('Unsupported protocol: ' + config.protocol)
        url = urlsplit(config.base_url)
        if url.scheme not in {'http', 'https'} or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('base_url must be an HTTP(S) endpoint without credentials, query or fragment')
        return config

    def request(self, prompt, max_output_tokens=2048):
        """Construct a request; credentials are returned only to the HTTP client."""
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError('Missing credential environment variable: ' + self.api_key_env)
        headers = {'Content-Type': 'application/json'}
        if self.protocol == 'openai_responses':
            suffix = '/responses'
            headers['Authorization'] = 'Bearer ' + key
            body = dict(model=self.model, input=prompt, max_output_tokens=max_output_tokens)
        elif self.protocol == 'openai_chat_completions':
            suffix = '/chat/completions'
            headers['Authorization'] = 'Bearer ' + key
            body = dict(model=self.model, messages=[dict(role='user', content=prompt)])
            body[self.max_tokens_field] = max_output_tokens
            if self.thinking is not None:
                body['thinking'] = {'type': self.thinking}
        else:
            suffix = '/messages'
            headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
            body = dict(model=self.model, messages=[dict(role='user', content=prompt)],
                        max_tokens=max_output_tokens)
        return self.base_url.rstrip('/') + suffix, headers, body


def generate(config, prompt, *, timeout=60, max_output_tokens=2048):
    import httpx
    url, headers, body = config.request(prompt, max_output_tokens)
    # Explicit endpoint only. Broken user-level proxy settings are not inherited.
    with httpx.Client(timeout=timeout, trust_env=config.trust_env) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
    if config.protocol == 'openai_responses':
        text = ''.join(part.get('text', '') for item in data.get('output', [])
                       for part in item.get('content', []) if part.get('type') == 'output_text')
    elif config.protocol == 'openai_chat_completions':
        text = data['choices'][0]['message']['content']
    else:
        text = ''.join(item.get('text', '') for item in data.get('content', []) if item.get('type') == 'text')
    return {'text': text, 'usage': data.get('usage'), 'response_id': data.get('id')}
