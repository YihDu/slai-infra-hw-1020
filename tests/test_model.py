import os
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from optimizer.model import ModelConfig,generate


class ModelTests(unittest.TestCase):
    def test_all_declared_protocols_use_config_and_decode_text(self):
        fixtures=[('openai_responses','/responses',dict(output=[dict(content=[dict(type='output_text',text='response')])])),
                  ('openai_chat_completions','/chat/completions',dict(choices=[dict(message=dict(content='response'))])),
                  ('anthropic_messages','/messages',dict(content=[dict(type='text',text='response')]))]
        for protocol,suffix,payload in fixtures:
            with self.subTest(protocol=protocol),patch.dict(os.environ,{'TEST_MODEL_KEY':'test-placeholder'}),patch('httpx.Client') as client:
                reply=MagicMock();reply.json.return_value=dict(payload,usage=dict(input_tokens=1,output_tokens=2))
                post=client.return_value.__enter__.return_value.post;post.return_value=reply
                config=ModelConfig(protocol,'https://example.invalid/custom/v1','specified-model','TEST_MODEL_KEY')
                response=generate(config,'prompt')
                self.assertEqual(response['text'],'response')
                self.assertEqual(post.call_args.args[0],'https://example.invalid/custom/v1'+suffix)
                self.assertEqual(post.call_args.kwargs['json']['model'],'specified-model')
                self.assertEqual(response['usage']['output_tokens'],2)

    def test_unsupported_protocol_and_missing_credentials_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text(json.dumps(dict(protocol='unknown',base_url='https://example.invalid',model='x',api_key_env='TEST_ABSENT_KEY')))
            with self.assertRaisesRegex(ValueError,'Unsupported protocol'):ModelConfig.load(p)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Missing credential'):
                ModelConfig('openai_responses','https://example.invalid','x','TEST_ABSENT_KEY').request('p')


if __name__=='__main__':unittest.main()
