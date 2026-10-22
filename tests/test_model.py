import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from optimizer.model import ModelAPIError, ModelConfig, generate

REAL_CLIENT = httpx.Client


@contextlib.contextmanager
def transport(handler):
    with patch('httpx.Client', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)):
        yield


class ModelTests(unittest.TestCase):
    def test_all_declared_protocols_honor_config_auth_and_parse_text(self):
        fixtures = [
            ('openai_responses', '/responses', dict(output=[dict(type='reasoning', summary=[]), dict(type='message', content=[dict(type='output_text', text='response')])])),
            ('openai_chat_completions', '/chat/completions', dict(choices=[dict(message=dict(content=[dict(type='text', text='response')]), finish_reason='stop')])),
            ('anthropic_messages', '/messages', dict(content=[dict(type='thinking', thinking='hidden'), dict(type='text', text='response')], stop_reason='end_turn'))]
        for protocol, suffix, payload in fixtures:
            with self.subTest(protocol=protocol), patch.dict(os.environ, {'TEST_MODEL_KEY': 'test-placeholder'}):
                def handler(request):
                    self.assertEqual(str(request.url), 'https://example.invalid/custom/v1' + suffix)
                    body = json.loads(request.content)
                    self.assertEqual(body['model'], 'specified-model')
                    if protocol == 'anthropic_messages':
                        self.assertEqual(request.headers['x-api-key'], 'test-placeholder')
                        self.assertEqual(request.headers['anthropic-version'], '2023-06-01')
                        self.assertEqual(body['max_tokens'], 2048)
                    else:
                        self.assertEqual(request.headers['Authorization'], 'Bearer test-placeholder')
                        self.assertNotIn('x-api-key', request.headers)
                    if protocol == 'openai_responses':
                        self.assertEqual(body['input'], 'prompt')
                        self.assertEqual(body['max_output_tokens'], 2048)
                    else:
                        self.assertEqual(body['messages'], [dict(role='user', content='prompt')])
                    return httpx.Response(200, json=dict(payload, id='fixture-response', usage=dict(input_tokens=1, output_tokens=2)))
                with transport(handler):
                    response = generate(ModelConfig(protocol, 'https://example.invalid/custom/v1', 'specified-model', 'TEST_MODEL_KEY'), 'prompt')
                self.assertEqual(response['text'], 'response')
                self.assertEqual(response['response_id'], 'fixture-response')
                self.assertEqual(response['usage']['output_tokens'], 2)
                self.assertEqual(response['request_count'], 1)

    def test_root_version_prefix_custom_prefix_and_full_endpoint(self):
        for protocol, suffix in [('openai_responses', '/responses'), ('openai_chat_completions', '/chat/completions'), ('anthropic_messages', '/messages')]:
            for base, expected in [('https://example.invalid', '/v1'+suffix), ('https://example.invalid/v1/', '/v1'+suffix), ('https://example.invalid/proxy/v9', '/proxy/v9'+suffix), ('https://example.invalid/v1'+suffix, '/v1'+suffix)]:
                with self.subTest(protocol=protocol, base=base):
                    self.assertEqual(ModelConfig(protocol, base, 'x', 'TEST_KEY').endpoint(), 'https://example.invalid'+expected)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            ModelConfig('openai_responses', 'https://example.invalid/v1/messages', 'x', 'TEST_KEY').endpoint()

    def test_auto_chat_token_parameter_negotiation_is_bounded_and_accounted(self):
        calls = []
        def handler(request):
            body = json.loads(request.content);calls.append(body)
            if 'max_completion_tokens' in body:
                return httpx.Response(400, json=dict(error=dict(code='unsupported_parameter', param='max_completion_tokens', message='Unsupported parameter')))
            self.assertEqual(body['max_tokens'], 77)
            return httpx.Response(200, json=dict(choices=[dict(message=dict(content='OK'))], usage=dict(prompt_tokens=3, completion_tokens=1)))
        with patch.dict(os.environ, {'TEST_KEY': 'test-placeholder'}), transport(handler):
            result = generate(ModelConfig('openai_chat_completions', 'https://example.invalid/v1', 'x', 'TEST_KEY'), 'p', max_output_tokens=77)
        self.assertEqual(result['request_count'], 2)
        self.assertEqual([a['status_code'] for a in result['http_attempts']], [400, 200])
        self.assertNotIn('max_completion_tokens', calls[1])
        self.assertIsNone(result['http_attempts'][0]['usage'])

    def test_does_not_retry_auth_quota_server_context_errors_or_explicit_field(self):
        for status, config_field, error in [(401, 'auto', dict(message='secret server body')),
            (429, 'auto', dict(message='rate limited')), (500, 'auto', dict(message='server error')),
            (400, 'auto', dict(param='max_completion_tokens', code='invalid_value', message='context length exceeded')),
            (400, 'max_completion_tokens', dict(param='max_completion_tokens', code='unsupported_parameter'))]:
            calls=[]
            def handler(request):
                calls.append(request);return httpx.Response(status, json=dict(error=error))
            with self.subTest(status=status, field=config_field), patch.dict(os.environ, {'TEST_KEY': 'test-placeholder'}), transport(handler):
                with self.assertRaises(ModelAPIError) as caught:
                    generate(ModelConfig('openai_chat_completions', 'https://example.invalid/v1', 'x', 'TEST_KEY', max_tokens_field=config_field), 'p')
                self.assertEqual(len(calls), 1)
                record=json.dumps(caught.exception.record())
                self.assertNotIn('secret server body', record)
                self.assertNotIn('test-placeholder', record)

    def test_truncated_and_empty_outputs_are_rejected_with_available_usage(self):
        fixtures=[('openai_responses',dict(status='incomplete',output=[])),
                  ('openai_chat_completions',dict(choices=[dict(message=dict(content='{}'),finish_reason='length')])),
                  ('anthropic_messages',dict(content=[dict(type='text',text='{}')],stop_reason='max_tokens')),
                  ('openai_responses',dict(output=[]))]
        for protocol,payload in fixtures:
            with self.subTest(protocol=protocol),patch.dict(os.environ,{'TEST_KEY':'test-placeholder'}),transport(lambda r:httpx.Response(200,json=dict(payload,usage=dict(input_tokens=2,output_tokens=4)))):
                with self.assertRaises(ModelAPIError) as caught:
                    generate(ModelConfig(protocol,'https://example.invalid/v1','x','TEST_KEY'),'p')
                self.assertEqual(caught.exception.usage['output_tokens'],4)

    def test_unsupported_protocol_and_missing_credentials_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text(json.dumps(dict(protocol='unknown',base_url='https://example.invalid',model='x',api_key_env='TEST_ABSENT_KEY')))
            with self.assertRaisesRegex(ValueError,'Unsupported protocol'):ModelConfig.load(p)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Missing credential'):
                ModelConfig('openai_responses','https://example.invalid','x','TEST_ABSENT_KEY').request('p')


if __name__=='__main__':unittest.main()
