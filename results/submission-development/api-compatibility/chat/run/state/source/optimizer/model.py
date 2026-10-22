"""Course model-config adapter. No personal agent settings or provider credentials."""
import json
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

ENDPOINTS = {'openai_responses': '/responses',
             'openai_chat_completions': '/chat/completions',
             'anthropic_messages': '/messages'}


class ModelAPIError(RuntimeError):
    """Safe diagnostic metadata; never includes response bodies or HTTP headers."""
    def __init__(self, reason, attempts, usage=None):
        super().__init__(reason)
        self.attempts = attempts
        self.request_count = len(attempts)
        self.usage = usage

    def record(self):
        return dict(error_type=type(self).__name__, reason=str(self),
                    http_attempts=self.attempts, request_count=self.request_count, usage=self.usage)


@dataclass(frozen=True)
class ModelConfig:
    protocol: str
    base_url: str
    model: str
    api_key_env: str
    max_tokens_field: str = 'auto'
    trust_env: bool = False
    thinking: str | None = None

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            data = json.load(stream)
        fields = ('protocol', 'base_url', 'model', 'api_key_env')
        if not isinstance(data, dict) or any(not isinstance(data.get(k), str) or not data[k].strip() for k in fields):
            raise ValueError('Model config requires nonempty strings: ' + ', '.join(fields))
        config = cls(**{k: data[k].strip() for k in fields},
                     max_tokens_field=data.get('max_tokens_field', 'auto'),
                     trust_env=data.get('trust_env', False), thinking=data.get('thinking'))
        config.validate()
        return config

    def validate(self):
        if self.protocol not in ENDPOINTS:
            raise ValueError('Unsupported protocol; use ' + ', '.join(ENDPOINTS))
        if not self.model.strip():
            raise ValueError('model must be nonempty')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', self.api_key_env):
            raise ValueError('api_key_env must name an environment variable')
        if self.thinking not in {None, 'enabled', 'disabled'}:
            raise ValueError('thinking must be enabled or disabled')
        if self.thinking is not None and self.protocol != 'openai_chat_completions':
            raise ValueError('thinking extension is supported only for Chat Completions')
        if self.max_tokens_field not in {'auto', 'max_tokens', 'max_completion_tokens'}:
            raise ValueError('Unsupported max_tokens_field')
        if not isinstance(self.trust_env, bool):
            raise ValueError('trust_env must be boolean')
        url = urlsplit(self.base_url)
        if (url.scheme not in {'http', 'https'} or not url.netloc or not url.hostname or
                url.username or url.password or url.query or url.fragment):
            raise ValueError('base_url must be an HTTP(S) endpoint without credentials, query or fragment')
        path = url.path.rstrip('/')
        for protocol, suffix in ENDPOINTS.items():
            if path.endswith(suffix) and protocol != self.protocol:
                raise ValueError('Endpoint path conflicts with configured protocol')

    def endpoint(self):
        self.validate()
        base = self.base_url.rstrip('/')
        suffix = ENDPOINTS[self.protocol]
        if urlsplit(base).path.endswith(suffix):
            return base
        # Provider roots use the standard /v1 namespace. Custom prefixes are exact.
        if not urlsplit(base).path:
            base += '/v1'
        return base + suffix

    def request(self, prompt, max_output_tokens=2048):
        """Credentials are returned only to the caller that sends the HTTP request."""
        url = self.endpoint()
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError('Missing credential environment variable: ' + self.api_key_env)
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('max_output_tokens must be a positive integer')
        headers = {'Content-Type': 'application/json'}
        if self.protocol == 'openai_responses':
            headers['Authorization'] = 'Bearer ' + key
            body = dict(model=self.model, input=prompt, max_output_tokens=max_output_tokens)
        elif self.protocol == 'openai_chat_completions':
            headers['Authorization'] = 'Bearer ' + key
            body = dict(model=self.model, messages=[dict(role='user', content=prompt)])
            field = 'max_completion_tokens' if self.max_tokens_field == 'auto' else self.max_tokens_field
            body[field] = max_output_tokens
            if self.thinking is not None:
                body['thinking'] = {'type': self.thinking}
        else:
            headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
            body = dict(model=self.model, messages=[dict(role='user', content=prompt)], max_tokens=max_output_tokens)
        return url, headers, body


def _legacy_token_field_required(response):
    """Retry only a rejected parameter schema, never auth/quota/server failures."""
    if response.status_code not in {400, 422}:
        return False
    try:
        error = response.json().get('error', {})
        if not isinstance(error, dict):
            return False
        message = str(error.get('message', '')).lower()
        return ((error.get('param') == 'max_completion_tokens' and
                 error.get('code') in {'unsupported_parameter', 'unknown_parameter'}) or
                ('max_completion_tokens' in message and any(word in message for word in
                 ('unsupported', 'unknown', 'unrecognized', 'not permitted', 'not allowed', 'extra inputs'))))
    except (ValueError, AttributeError):
        return False


def _text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p['text'] for p in content if isinstance(p, dict) and
                       p.get('type') in {'text', 'output_text'} and isinstance(p.get('text'), str))
    return ''


def _decode(config, data, attempts):
    usage = data.get('usage') if isinstance(data.get('usage'), dict) else None
    if config.protocol == 'openai_responses':
        if data.get('status') in {'incomplete', 'failed', 'cancelled', 'queued', 'in_progress'}:
            raise ModelAPIError('Responses API did not return a completed response', attempts, usage)
        text = ''.join(_text_content(item.get('content')) for item in data.get('output', [])
                       if isinstance(item, dict) and item.get('type', 'message') == 'message')
        if not text:
            text = _text_content(data.get('output_text'))
    elif config.protocol == 'openai_chat_completions':
        choices = data.get('choices') or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        if first.get('finish_reason') in {'length', 'content_filter'}:
            raise ModelAPIError('Chat completion was truncated or filtered', attempts, usage)
        message = first.get('message') or {}
        text = _text_content(message.get('content')) if isinstance(message, dict) else ''
    else:
        if data.get('stop_reason') in {'max_tokens', 'refusal', 'pause_turn'}:
            raise ModelAPIError('Anthropic message was truncated, refused or paused', attempts, usage)
        text = _text_content(data.get('content'))
    if not text.strip():
        raise ModelAPIError('Model returned no usable text; text/JSON generation is required', attempts, usage)
    return dict(text=text, usage=usage, response_id=data.get('id'),
                request_count=len(attempts), http_attempts=attempts)


def generate(config, prompt, *, timeout=60, max_output_tokens=2048):
    import httpx
    url, headers, body = config.request(prompt, max_output_tokens)
    attempts = []
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=timeout, trust_env=config.trust_env, follow_redirects=False) as client:
        for index in range(2):
            attempt = dict(status_code=None, token_limit_field=next(
                (f for f in ('max_completion_tokens', 'max_tokens', 'max_output_tokens') if f in body), None))
            attempts.append(attempt)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelAPIError('Model request deadline exceeded', attempts)
            try:
                response = client.post(url, headers=headers, json=body, timeout=remaining)
            except httpx.HTTPError as exc:
                attempt['error_type'] = type(exc).__name__
                raise ModelAPIError('Model network request failed: ' + type(exc).__name__, attempts) from None
            attempt['status_code'] = response.status_code
            if (index == 0 and config.protocol == 'openai_chat_completions' and
                    config.max_tokens_field == 'auto' and _legacy_token_field_required(response)):
                attempt['retry_reason'] = 'Endpoint rejects max_completion_tokens; retry max_tokens'
                attempt['usage'] = None
                body = dict(body)
                body['max_tokens'] = body.pop('max_completion_tokens')
                continue
            if not 200 <= response.status_code < 300:
                raise ModelAPIError('Model endpoint returned HTTP ' + str(response.status_code), attempts)
            try:
                data = response.json()
            except ValueError:
                raise ModelAPIError('Model endpoint returned invalid JSON', attempts) from None
            if not isinstance(data, dict):
                raise ModelAPIError('Model endpoint response must be a JSON object', attempts)
            return _decode(config, data, attempts)
    raise ModelAPIError('Model request failed', attempts)
