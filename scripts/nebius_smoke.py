"""One-shot Nebius API/native-tool smoke test; stdlib only, no robot imports.

Only reads the local .env; never falls back to environment/chat credentials.
A reserved result file prevents accidental reruns, including after interruptions.
"""
import argparse
import http.client
import json
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://api.tokenfactory.nebius.com/v1'
MODEL = 'nvidia/Nemotron-3_5-Lightning'
OUT = ROOT / 'outputs/nebius-smoke-v1/result.json'
MAX_POSTS = 3  # Stricter than the user's maximum of ten paid requests.
MAX_TOKENS = 1024
MAX_BODY_BYTES = 16384
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PRICE = {'input_usd_per_million': 0.06, 'output_usd_per_million': 0.24,
         'checked_date': '2026-10-31',
         'source': 'https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference'}
TOOL = {'type': 'function', 'function': {
    'name': 'add_integers', 'description': 'Add two integers locally without side effects.',
    'parameters': {'type': 'object', 'properties': {
        'a': {'type': 'integer', 'minimum': -1000, 'maximum': 1000},
        'b': {'type': 'integer', 'minimum': -1000, 'maximum': 1000}},
        'required': ['a', 'b'], 'additionalProperties': False}}}


class SmokeError(Exception):
    """Only static, non-secret error codes may be raised here."""


def load_config(path):
    if not path.is_file():
        raise SmokeError('local_env_missing')
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, sep, value = line.partition('=')
        name = name.strip()
        if name not in {'NEBIUS_BASE_URL', 'NEBIUS_MODEL', 'NEBIUS_API_KEY'}:
            continue
        if not sep or name in values:
            raise SmokeError('invalid_or_duplicate_nebius_config')
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    if values.get('NEBIUS_BASE_URL') != BASE_URL or values.get('NEBIUS_MODEL') != MODEL:
        raise SmokeError('endpoint_or_model_differs_from_registered_smoke')
    key = values.get('NEBIUS_API_KEY', '')
    if key and not re.fullmatch(r'[A-Za-z0-9._~+/=-]{20,4096}', key):
        raise SmokeError('invalid_key_format')
    return key


def ensure_env_ignored():
    ignored = subprocess.run(['git', 'check-ignore', '-q', '--', '.env'], cwd=ROOT,
                             capture_output=True).returncode == 0
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', '.env'], cwd=ROOT,
                             capture_output=True).returncode == 0
    if not ignored or tracked:
        raise SmokeError('env_must_be_ignored_and_untracked')


def sanitize(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]') if key else value
    if isinstance(value, dict):
        return {sanitize(k, key): sanitize(v, key) for k, v in value.items()
                if k.lower() not in {'authorization', 'api_key', 'nebius_api_key'}}
    if isinstance(value, list):
        return [sanitize(v, key) for v in value]
    return value


def numeric_usage(value):
    """Keep actual numeric usage only; never provider text/error payloads."""
    if not isinstance(value, dict):
        return None
    result = {}
    for k in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens',
              'output_tokens', 'prompt_tokens_details', 'completion_tokens_details',
              'cached_tokens', 'reasoning_tokens', 'accepted_prediction_tokens',
              'rejected_prediction_tokens', 'audio_tokens']:
        v = value.get(k)
        if isinstance(v, dict):
            result[k] = numeric_usage(v)
        elif type(v) is int and v >= 0:
            result[k] = v
    return result or None


def usage_cost(usage):
    if not usage:
        return None
    inp = usage.get('prompt_tokens', usage.get('input_tokens'))
    out = usage.get('completion_tokens', usage.get('output_tokens'))
    if type(inp) is not int or type(out) is not int:
        return None
    return (inp * PRICE['input_usd_per_million'] + out * PRICE['output_usd_per_million']) / 1e6


def validate_tool_call(message):
    calls = message.get('tool_calls')
    if not isinstance(calls, list) or len(calls) != 1:
        raise SmokeError('expected_one_native_tool_call')
    call = calls[0]
    if not isinstance(call, dict) or call.get('type') != 'function':
        raise SmokeError('invalid_native_tool_type')
    call_id = call.get('id')
    if not isinstance(call_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', call_id):
        raise SmokeError('invalid_native_tool_call_id')
    fn = call.get('function', {})
    if not isinstance(fn, dict) or fn.get('name') != 'add_integers':
        raise SmokeError('unexpected_tool_name')
    args_raw = fn.get('arguments')
    if not isinstance(args_raw, str) or len(args_raw) > 1024:
        raise SmokeError('invalid_tool_arguments_encoding')
    try:
        args = json.loads(args_raw)
    except (ValueError, TypeError):
        raise SmokeError('invalid_tool_arguments_json') from None
    if not isinstance(args, dict) or set(args) != {'a', 'b'}:
        raise SmokeError('unexpected_tool_arguments')
    if any(type(v) is not int or abs(v) > 1000 for v in args.values()):
        raise SmokeError('tool_arguments_out_of_bounds')
    if args != {'a': 17, 'b': 25}:
        raise SmokeError('tool_arguments_do_not_match_smoke_prompt')
    result = {'sum': args['a'] + args['b']}
    assistant = {'role': 'assistant', 'content': message.get('content'),
                 'tool_calls': [{'id': call_id, 'type': 'function',
                                 'function': {'name': 'add_integers', 'arguments': args_raw}}]}
    tool_reply = {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(result)}
    return assistant, tool_reply, result


def assistant_message(data):
    choices = data.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise SmokeError('invalid_completion_choices')
    message = choices[0].get('message')
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        raise SmokeError('invalid_assistant_message')
    return message, choices[0].get('finish_reason')


class Smoke:
    def __init__(self, key, output=OUT):
        self.key, self.output = key, output
        self.report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'configured_model': MODEL, 'base_url': BASE_URL, 'api_connection': False,
            'available_model_id': None, 'chat_completion_success': False,
            'native_tool_calling_success': False, 'paid_requests_attempted': 0,
            'request_limit': MAX_POSTS, 'max_output_tokens_per_request': MAX_TOKENS,
            'retries': 0, 'price_snapshot': PRICE, 'native_tool_schema': TOOL, 'requests': [],
            'budget_note': 'USD 1 local estimate target, not a provider billing hard limit. Unknown usage is not zero cost.',
            'planning_estimate_usd': MAX_POSTS * (MAX_BODY_BYTES * 0.06 + MAX_TOKENS * 0.24) / 1e6,
            'planning_estimate_basis': 'Conservative local allowance: input tokens approximated by maximum serialized UTF-8 bytes; provider accounting can differ.'}

    def reserve(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.output.open('x', encoding='utf-8') as f:
                json.dump(self.report, f, indent=2)
        except FileExistsError:
            raise SmokeError('existing_run_preserved_no_automatic_rerun') from None

    def save(self):
        posts = [r for r in self.report['requests'] if r['method'] == 'POST']
        known = [r['estimated_cost_usd'] for r in posts if r.get('estimated_cost_usd') is not None]
        self.report['known_usage_estimated_cost_usd'] = sum(known)
        self.report['requests_with_unknown_cost'] = len(posts) - len(known)
        self.report['total_estimated_cost_usd'] = sum(known) if len(known) == len(posts) else None
        tmp = self.output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(sanitize(self.report, self.key), indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        tmp.replace(self.output)

    def request(self, label, method, payload=None):
        if method not in {'GET', 'POST'}:
            raise SmokeError('unexpected_http_method')
        body = json.dumps(payload).encode() if payload is not None else None
        if body is not None and len(body) > MAX_BODY_BYTES:
            raise SmokeError('request_body_budget_exceeded')
        if method == 'POST':
            if self.report['paid_requests_attempted'] >= MAX_POSTS:
                raise SmokeError('paid_request_cap_reached')
            self.report['paid_requests_attempted'] += 1
        row = {'label': label, 'method': method, 'status': 'attempting', 'usage': None,
               'estimated_cost_usd': None}
        self.report['requests'].append(row)
        self.save()  # Count before sending, including uncertain outcomes/interruption.
        start = time.perf_counter()
        conn = http.client.HTTPSConnection('api.tokenfactory.nebius.com', timeout=60,
                                          context=ssl.create_default_context())
        try:
            path = '/v1/models' if method == 'GET' else '/v1/chat/completions'
            conn.request(method, path, body=body, headers={'Authorization': 'Bearer '+self.key,
                         'Content-Type': 'application/json', 'Accept': 'application/json'})
            response = conn.getresponse()
            row['http_status'] = response.status
            # http.client follows no redirects and makes no retries.
            if response.status != 200:
                row['status'] = 'http_error'
                raise SmokeError('api_http_error')
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise SmokeError('response_size_limit')
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                raise SmokeError('response_not_json') from None
            if not isinstance(data, dict):
                raise SmokeError('response_not_object')
            row['usage'] = numeric_usage(data.get('usage'))
            row['estimated_cost_usd'] = usage_cost(row['usage']) if method == 'POST' else 0.0
            row['response_model'] = str(data.get('model', ''))[:200]
            row['response_id'] = str(data.get('id', ''))[:200]
            row['status'] = 'received'
            if method == 'POST' and data.get('model') != MODEL:
                raise SmokeError('response_model_mismatch')
            return data
        except SmokeError:
            raise
        except Exception:
            row['status'] = 'transport_error'
            raise SmokeError('transport_error_no_retry') from None
        finally:
            row['latency_seconds'] = time.perf_counter() - start
            conn.close()
            self.save()

    def chat(self, label, messages, **extra):
        return self.request(label, 'POST', {'model': MODEL, 'messages': messages,
            'max_tokens': MAX_TOKENS, 'temperature': 0, 'stream': False, **extra})

    def run(self):
        self.reserve()
        started = time.perf_counter()
        try:
            data = self.request('models', 'GET')
            models = data.get('data')
            if not isinstance(models, list):
                raise SmokeError('invalid_models_response')
            ids = [m['id'] for m in models if isinstance(m, dict) and isinstance(m.get('id'), str)]
            self.report['api_connection'] = True
            self.report['available_nemotron_model_ids'] = [x for x in ids if 'nemotron' in x.lower()]
            if MODEL not in ids:
                raise SmokeError('configured_model_not_in_authenticated_catalog')
            self.report['available_model_id'] = MODEL
            data = self.chat('chat_completion', [{'role': 'user', 'content': 'Reply exactly NEBIUS_SMOKE_OK.'}])
            msg, finish = assistant_message(data)
            self.report['chat_reply'] = str(msg.get('content', ''))[:1024]
            self.report['requests'][-1]['finish_reason'] = finish
            if finish != 'stop' or msg.get('tool_calls') or not isinstance(msg.get('content'), str) or msg['content'].strip() != 'NEBIUS_SMOKE_OK':
                raise SmokeError('chat_completion_smoke_answer_failed')
            self.report['chat_completion_success'] = True
            messages = [{'role': 'user', 'content': 'Use add_integers with a=17 and b=25. After the tool result, reply with only the numeric sum.'}]
            data = self.chat('native_tool_call', messages, tools=[TOOL],
                tool_choice={'type': 'function', 'function': {'name': 'add_integers'}})
            msg, finish = assistant_message(data)
            self.report['requests'][-1]['finish_reason'] = finish
            if finish != 'tool_calls':
                raise SmokeError('completion_did_not_finish_with_tool_calls')
            assistant, reply, result = validate_tool_call(msg)
            self.report['native_tool_exchange'] = {'assistant': assistant, 'tool': reply, 'local_result': result}
            data = self.chat('tool_result_completion', messages+[assistant, reply], tools=[TOOL], tool_choice='none')
            msg, finish = assistant_message(data)
            self.report['requests'][-1]['finish_reason'] = finish
            self.report['final_reply'] = str(msg.get('content', ''))[:1024]
            if finish != 'stop' or msg.get('tool_calls') or not isinstance(msg.get('content'), str) or msg['content'].strip() != '42':
                raise SmokeError('tool_result_final_answer_failed')
            self.report['native_tool_calling_success'] = True
            self.report['status'] = 'Ready for Agent Integration'
        except SmokeError as exc:
            self.report['status'] = 'failed'
            self.report['failure_code'] = str(exc)  # Static local error code only.
        except KeyboardInterrupt:
            self.report['status'] = 'interrupted_no_retry'
        except Exception:
            self.report['status'] = 'failed'
            self.report['failure_code'] = 'unexpected_error_details_suppressed'
        finally:
            self.report['elapsed_seconds'] = time.perf_counter() - started
            self.save()
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-config', action='store_true', help='No network; report only whether local key is configured.')
    args = parser.parse_args()
    try:
        ensure_env_ignored()
        key = load_config(ROOT / '.env')
        if args.check_config or not key:
            print(json.dumps({'status': 'configured_not_tested' if key else 'awaiting_fresh_key_input',
                              'key_configured': bool(key), 'paid_requests': 0, 'network_requests': 0}))
            return 0 if key else 2
        report = Smoke(key).run()
        print(json.dumps(sanitize({'status': report['status'], 'report': str(OUT),
              'available_model_id': report['available_model_id'],
              'chat_completion_success': report['chat_completion_success'],
              'native_tool_calling_success': report['native_tool_calling_success'],
              'paid_requests_attempted': report['paid_requests_attempted'],
              'requests': report['requests'], 'total_estimated_cost_usd': report['total_estimated_cost_usd'],
              'failure_code': report.get('failure_code')}, key), indent=2))
        return 0 if report['status'] == 'Ready for Agent Integration' else 1
    except SmokeError as exc:
        print(json.dumps({'status': 'not_run', 'reason': str(exc)}))
        return 2
    except Exception:
        print(json.dumps({'status': 'not_run', 'reason': 'local_setup_error_details_suppressed'}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
