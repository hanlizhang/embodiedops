"""Offline security/contract tests only; fixtures are NOT API success evidence."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location('nebius_smoke', Path(__file__).resolve().parents[1] / 'scripts/nebius_smoke.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
FIXTURE_KEY = 'offline_fixture_token_never_a_real_key'


def tool_message(arguments='{"a":17,"b":25}', call_id='call_offline_1'):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': 'add_integers', 'arguments': arguments}}]}


def completion(message, finish='stop'):
    return {'model': smoke.MODEL, 'choices': [{'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}


def connection(payload=None, status=200, error=None):
    conn = MagicMock()
    if error:
        conn.request.side_effect = error
    response = conn.getresponse.return_value
    response.status = status
    response.read.return_value = json.dumps(payload).encode()
    return conn


class NebiusSmokeTests(unittest.TestCase):
    def test_config_has_no_environment_credential_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / '.env'
            env.write_text(f'NEBIUS_BASE_URL={smoke.BASE_URL}\nNEBIUS_MODEL={smoke.MODEL}\nNEBIUS_API_KEY=\n')
            with patch.dict(os.environ, {'NEBIUS_API_KEY': FIXTURE_KEY}):
                self.assertEqual(smoke.load_config(env), '')
            env.write_text(f'NEBIUS_BASE_URL=https://example.invalid/v1\nNEBIUS_MODEL={smoke.MODEL}\nNEBIUS_API_KEY={FIXTURE_KEY}\n')
            with self.assertRaisesRegex(smoke.SmokeError, 'endpoint_or_model'):
                smoke.load_config(env)

    def test_duplicate_or_header_injection_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / '.env'
            prefix = f'NEBIUS_BASE_URL={smoke.BASE_URL}\nNEBIUS_MODEL={smoke.MODEL}\n'
            for body in [f'NEBIUS_API_KEY={FIXTURE_KEY}\nNEBIUS_API_KEY=another', 'NEBIUS_API_KEY=contains spaces not a token']:
                env.write_text(prefix + body)
                with self.assertRaises(smoke.SmokeError):
                    smoke.load_config(env)

    def test_only_actual_single_bounded_named_call_is_accepted(self):
        bad_args = ['{"a":true,"b":25}', '{"a":17.0,"b":25}', '{"a":1001,"b":25}',
                    '{"a":17,"b":25,"shell":"bad"}', '{"a":18,"b":25}', '[]', 'not JSON']
        for args in bad_args:
            with self.subTest(args=args), self.assertRaises(smoke.SmokeError):
                smoke.validate_tool_call(tool_message(args))
        for call_id in ['', 'bad\nheader', 'a' * 201]:
            with self.subTest(call_id=call_id), self.assertRaises(smoke.SmokeError):
                smoke.validate_tool_call(tool_message(call_id=call_id))
        for message in [{'content': '{"tool":"add_integers"}'}, {'tool_calls': []},
                        {'tool_calls': tool_message()['tool_calls'] * 2}]:
            with self.assertRaises(smoke.SmokeError):
                smoke.validate_tool_call(message)
        message = tool_message()
        message['tool_calls'][0]['function']['name'] = 'run_shell'
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_tool_call(message)

    def test_tool_result_preserves_actual_call_id(self):
        assistant, reply, result = smoke.validate_tool_call(tool_message(call_id='provider_call_123'))
        self.assertEqual(reply['tool_call_id'], 'provider_call_123')
        self.assertEqual(assistant['tool_calls'][0]['id'], reply['tool_call_id'])
        self.assertEqual(result, {'sum': 42})
        self.assertEqual(json.loads(reply['content']), result)

    def test_usage_allowlist_and_unknown_cost(self):
        self.assertIsNone(smoke.usage_cost(None))
        self.assertIsNone(smoke.usage_cost({'prompt_tokens': 12}))
        usage = smoke.numeric_usage({'prompt_tokens': 100, 'completion_tokens': 10,
                    'total_tokens': True, 'untrusted_text': FIXTURE_KEY,
                    'completion_tokens_details': {'reasoning_tokens': 3, 'text': FIXTURE_KEY}})
        self.assertNotIn('total_tokens', usage)
        self.assertNotIn('untrusted_text', usage)
        self.assertEqual(usage['completion_tokens_details'], {'reasoning_tokens': 3})
        self.assertAlmostEqual(smoke.usage_cost(usage), 0.0000084)

    def test_redaction_removes_header_fields_and_echoed_key(self):
        safe = smoke.sanitize({'Authorization': 'Bearer ' + FIXTURE_KEY, 'NEBIUS_API_KEY': FIXTURE_KEY,
                               'nested': [{'reply': 'echo ' + FIXTURE_KEY}]}, FIXTURE_KEY)
        self.assertNotIn('Authorization', safe)
        self.assertNotIn('NEBIUS_API_KEY', safe)
        self.assertNotIn(FIXTURE_KEY, json.dumps(safe))
        self.assertIn('[REDACTED]', json.dumps(safe))

    def test_offline_fixture_roundtrip_is_three_posts_with_matching_tool_id(self):
        responses = [connection({'data': [{'id': smoke.MODEL}]}),
            connection(completion({'role': 'assistant', 'content': 'NEBIUS_SMOKE_OK'})),
            connection(completion(tool_message(), 'tool_calls')),
            connection(completion({'role': 'assistant', 'content': '42'}))]
        with tempfile.TemporaryDirectory() as temp, patch.object(smoke.http.client, 'HTTPSConnection', side_effect=responses) as factory:
            output = Path(temp) / 'offline-only.json'
            report = smoke.Smoke(FIXTURE_KEY, output).run()
            self.assertEqual(report['status'], 'Ready for Agent Integration')
            self.assertEqual(report['paid_requests_attempted'], 3)
            self.assertEqual(factory.call_count, 4)
            methods = [c.request.call_args.args[0] for c in responses]
            self.assertEqual(methods, ['GET', 'POST', 'POST', 'POST'])
            final = json.loads(responses[-1].request.call_args.kwargs['body'])
            self.assertEqual(final['messages'][-1]['tool_call_id'], final['messages'][-2]['tool_calls'][0]['id'])
            self.assertEqual(final['tool_choice'], 'none')
            self.assertLessEqual(final['max_tokens'], smoke.MAX_TOKENS)
            trace = output.read_text(encoding='utf-8')
            self.assertNotIn(FIXTURE_KEY, trace)
            self.assertNotIn('Authorization', trace)
            self.assertLess(report['total_estimated_cost_usd'], 1)

    def test_failed_paid_request_is_not_retried_and_cost_is_unknown(self):
        responses = [connection({'data': [{'id': smoke.MODEL}]}), connection(error=TimeoutError(FIXTURE_KEY))]
        with tempfile.TemporaryDirectory() as temp, patch.object(smoke.http.client, 'HTTPSConnection', side_effect=responses) as factory:
            output = Path(temp) / 'offline-only.json'
            report = smoke.Smoke(FIXTURE_KEY, output).run()
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(report['paid_requests_attempted'], 1)
            self.assertEqual(report['failure_code'], 'transport_error_no_retry')
            self.assertEqual(report['requests_with_unknown_cost'], 1)
            self.assertIsNone(report['total_estimated_cost_usd'])
            self.assertNotIn(FIXTURE_KEY, output.read_text())

    def test_paid_request_cap_blocks_before_transport(self):
        responses = [connection({'model': smoke.MODEL}) for _ in range(smoke.MAX_POSTS)]
        with tempfile.TemporaryDirectory() as temp, patch.object(smoke.http.client, 'HTTPSConnection', side_effect=responses) as factory:
            runner = smoke.Smoke(FIXTURE_KEY, Path(temp) / 'offline-only.json')
            runner.reserve()
            for _ in range(smoke.MAX_POSTS):
                runner.request('offline-test', 'POST', {})
            with self.assertRaisesRegex(smoke.SmokeError, 'paid_request_cap_reached'):
                runner.request('offline-test', 'POST', {})
            self.assertEqual(factory.call_count, smoke.MAX_POSTS)

    def test_existing_result_prevents_rerun_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(smoke.http.client, 'HTTPSConnection') as factory:
            output = Path(temp) / 'offline-only.json'
            output.write_text('preserve this prior attempt')
            with self.assertRaisesRegex(smoke.SmokeError, 'existing_run_preserved'):
                smoke.Smoke(FIXTURE_KEY, output).run()
            self.assertEqual(output.read_text(), 'preserve this prior attempt')
            factory.assert_not_called()

    def test_redirect_is_failure_not_followed(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(smoke.http.client, 'HTTPSConnection', return_value=connection(status=302)) as factory:
            report = smoke.Smoke(FIXTURE_KEY, Path(temp) / 'offline-only.json').run()
            self.assertEqual(report['failure_code'], 'api_http_error')
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(report['paid_requests_attempted'], 0)


if __name__ == '__main__':
    unittest.main()
