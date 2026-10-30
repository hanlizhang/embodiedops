"""New agent-loop contract tests. Offline fixtures are never real API evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import nemotron_agent_demo as demo
import nemotron_case_tools as case_tools

KEY = 'offline_agent_fixture_never_a_real_key'


def call(name, args=None, ident='call_fixture'):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': ident, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(args or {'case': 'seed64'})}}]}


def final(text='Fixture explanation only.'):
    return {'role': 'assistant', 'content': text}


class FakeBackend:
    schemas = [{'type': 'function', 'function': {'name': n,
                'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
               for n in ['inspect_case', 'inspect_trajectory', 'compare_candidates',
                         'run_bounded_verification', 'check_acceptance']]

    def __init__(self):
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        if name == 'check_acceptance':
            return {'accepted': False, 'reasons': ['per_seed_success_regression'], 'source': args.get('source', 'archived')}
        if name == 'run_bounded_verification':
            return {'source': 'new', 'episodes_attempted': 2, 'valid': True}
        if name in {'inspect_case', 'inspect_trajectory', 'compare_candidates'}:
            return {'source': 'archived', 'observation': 'fixture:' + name, 'candidate_success': False}
        raise ValueError('Do not print exception body: ' + KEY)


class FixtureClient:
    def __init__(self, sequence):
        self.sequence = iter(sequence)
        self.inputs = []

    def factory(self, key, trace, save):
        self.trace, self.save = trace, save
        return self

    def complete(self, messages, schemas, final_only=False):
        self.inputs.append({'messages': copy.deepcopy(messages), 'final_only': final_only})
        self.trace['requests'].append({'index': len(self.inputs), 'estimated_cost_usd': 0.0,
            'latency_seconds': 0.0, 'usage': {'prompt_tokens': 1, 'completion_tokens': 1,
                                             'total_tokens': 2, 'completion_tokens_details': {'reasoning_tokens': 1}}})
        msg = next(self.sequence)
        return msg, 'tool_calls' if msg.get('tool_calls') else 'stop'


def fake_connection(payload=None, error=None, status=200):
    conn = MagicMock()
    if error:
        conn.request.side_effect = error
    conn.getresponse.return_value.status = status
    conn.getresponse.return_value.read.return_value = json.dumps(payload).encode()
    return conn


class NemotronAgentTests(unittest.TestCase):
    def run_loop(self, sequence):
        backend = FakeBackend()
        client = FixtureClient(sequence)
        with tempfile.TemporaryDirectory() as temp, patch.object(demo, 'archived_case_summary', return_value={}):
            runner = demo.AgentLoop(KEY, backend, Path(temp), client_factory=client.factory)
            result = runner.run()
            trace = json.loads((Path(temp) / 'trace.json').read_text(encoding='utf-8'))
            serialized = (Path(temp) / 'result.json').read_text(encoding='utf-8')
            self.assertNotIn(KEY, serialized)
        return result, trace, backend, client

    def test_only_one_unique_native_call_with_bounded_json_is_parsed(self):
        for message in [final('{"tool":"inspect_case"}'), {'tool_calls': []},
                        {'tool_calls': call('inspect_case')['tool_calls'] * 2}]:
            with self.assertRaises(demo.SmokeError):
                demo.parse_tool_call(message, set())
        for raw in ['[]', '{"case":"seed64","case":"seed64"}', '{', ' ' * 2049]:
            message = call('inspect_case')
            message['tool_calls'][0]['function']['arguments'] = raw
            with self.assertRaises(demo.SmokeError):
                demo.parse_tool_call(message, set())
        seen = set()
        self.assertEqual(demo.parse_tool_call(call('inspect_case'), seen)[0], 'call_fixture')
        with self.assertRaisesRegex(demo.SmokeError, 'repeated'):
            demo.parse_tool_call(call('inspect_case'), seen)

    def test_model_selected_order_and_every_tool_result_roundtrips(self):
        sequence = [call('compare_candidates', ident='first'),
                    call('inspect_trajectory', ident='second'),
                    call('check_acceptance', ident='third'), final('ACCEPT REPAIR: deliberately wrong fixture prose')]
        result, trace, backend, client = self.run_loop(sequence)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([x[0] for x in backend.calls], ['compare_candidates', 'inspect_trajectory', 'check_acceptance'])
        self.assertEqual(result['tool_sequence'], [x[0] for x in backend.calls])
        self.assertFalse(result['independent_acceptance']['accepted'])
        self.assertTrue(result['all_tool_results_returned'])
        for i, event in enumerate(trace['events'], 1):
            sent = client.inputs[i]['messages'][-1]
            self.assertEqual(sent['role'], 'tool')
            self.assertEqual(sent['tool_call_id'], event['tool_call_id'])
            self.assertEqual(json.loads(sent['content']), event['result'])
            self.assertEqual(event['returned_to_model_request'], i + 1)

    def test_eight_decisions_have_one_final_no_tools_synthesis(self):
        sequence = [call('inspect_case', ident='decision_0'), call('check_acceptance', ident='decision_1')]
        sequence += [call('inspect_trajectory', ident=f'decision_{i}') for i in range(2, 8)]
        sequence += [final()]
        result, trace, backend, client = self.run_loop(sequence)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['decision_steps'], 8)
        self.assertEqual(result['api_requests'], 9)
        self.assertEqual(len(backend.calls), 8)
        self.assertTrue(client.inputs[-1]['final_only'])
        self.assertTrue(result['all_tool_results_returned'])

    def test_ninth_tool_call_is_never_dispatched(self):
        sequence = [call('inspect_case', ident='decision_0'), call('check_acceptance', ident='decision_1')]
        sequence += [call('inspect_trajectory', ident=f'decision_{i}') for i in range(2, 9)]
        result, trace, backend, client = self.run_loop(sequence)
        self.assertEqual(result['failure_code'], 'tool_call_outside_decision_budget')
        self.assertEqual(len(backend.calls), 8)
        self.assertLessEqual(result['api_requests'], 10)

    def test_final_without_gate_is_incomplete_not_success(self):
        result, _, _, _ = self.run_loop([call('inspect_case'), final()])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_code'], 'agent_stopped_without_independent_acceptance')

    def test_new_verification_invalidates_cached_acceptance(self):
        result, _, _, _ = self.run_loop([call('inspect_case', ident='i'),
            call('check_acceptance', ident='a'), call('run_bounded_verification', ident='v'), final()])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_code'], 'agent_stopped_without_independent_acceptance')
        self.assertIsNone(result['independent_acceptance'])

    def test_tool_error_is_returned_without_secret_exception_body(self):
        result, trace, _, client = self.run_loop([call('run_shell', ident='bad'), final()])
        error = json.loads(client.inputs[1]['messages'][-1]['content'])
        self.assertEqual(error['tool_error'], 'tool_rejected_or_failed')
        self.assertFalse(error['retry_allowed'])
        self.assertNotIn(KEY, json.dumps(trace))
        self.assertEqual(result['status'], 'failed')

    def test_client_enforces_request_and_input_limits_before_transport(self):
        with patch.object(demo.http.client, 'HTTPSConnection') as factory:
            client = demo.AgentClient(KEY, {'requests': [{}] * demo.MAX_REQUESTS, 'responses': []}, lambda: None)
            with self.assertRaisesRegex(demo.SmokeError, 'api_request_limit'):
                client.complete([], [])
            client = demo.AgentClient(KEY, {'requests': [], 'responses': []}, lambda: None)
            with self.assertRaisesRegex(demo.SmokeError, 'input_size_limit'):
                client.complete([{'role': 'user', 'content': 'x' * demo.MAX_INPUT_BYTES}], [])
            factory.assert_not_called()

    def test_client_has_no_retry_or_redirect_and_records_unknown_usage(self):
        for connection in [fake_connection(error=TimeoutError(KEY)), fake_connection(status=302)]:
            trace = {'requests': [], 'responses': []}
            with patch.object(demo.http.client, 'HTTPSConnection', return_value=connection) as factory:
                client = demo.AgentClient(KEY, trace, lambda: None)
                with self.assertRaises(demo.SmokeError) as err:
                    client.complete([], [])
                self.assertNotIn(KEY, str(err.exception))
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(connection.request.call_count, 1)
                self.assertEqual(len(trace['requests']), 1)
                self.assertIsNone(trace['requests'][0]['estimated_cost_usd'])
                self.assertNotIn(KEY, json.dumps(trace))

    def test_client_sends_native_schemas_auto_choice_and_includes_reasoning_usage(self):
        data = {'id': 'offline_response', 'model': demo.MODEL,
                'choices': [{'message': call('inspect_case'), 'finish_reason': 'tool_calls'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                          'completion_tokens_details': {'reasoning_tokens': 10}}}
        connection = fake_connection(data)
        trace = {'requests': [], 'responses': []}
        with patch.object(demo.http.client, 'HTTPSConnection', return_value=connection):
            msg, finish = demo.AgentClient(KEY, trace, lambda: None).complete([], FakeBackend.schemas)
        body = json.loads(connection.request.call_args.kwargs['body'])
        self.assertEqual(body['tool_choice'], 'auto')
        self.assertFalse(body['parallel_tool_calls'])
        self.assertEqual(body['tools'], FakeBackend.schemas)
        self.assertEqual(body['max_tokens'], demo.MAX_OUTPUT_TOKENS)
        self.assertEqual(trace['requests'][0]['usage']['completion_tokens_details']['reasoning_tokens'], 10)
        self.assertAlmostEqual(trace['requests'][0]['estimated_cost_usd'], (100 * .06 + 20 * .24) / 1e6)
        self.assertEqual(finish, 'tool_calls')


    def test_real_case_tools_reject_paths_shell_and_acceptance_overrides(self):
        for name, args in [
            ('run_shell', {'case': 'seed64'}),
            ('inspect_case', {'case': 'seed64', 'path': '.env'}),
            ('inspect_case', {'case': 'seed48'}),
            ('check_acceptance', {'case': 'seed64', 'source': 'archived', 'accepted': True}),
            ('run_bounded_verification', {'case': 'seed64', 'rationale': 'x' * 501}),
            ('inspect_trajectory', {'case': 'seed64', 'model': 'other', 'segment': 'all'}),
        ]:
            with self.subTest(tool=name), self.assertRaises(ValueError):
                case_tools.validate_arguments(name, args)
        with tempfile.TemporaryDirectory() as temp, patch.object(case_tools.subprocess, 'run') as process:
            backend = case_tools.CaseTools.__new__(case_tools.CaseTools)
            backend.verification_attempted = True
            backend.physical = Path(temp)
            with self.assertRaises(ValueError):
                backend.run_bounded_verification('seed64', 'No second pair may run.')
            process.assert_not_called()

    def test_independent_gate_rejects_archived_regression_and_stale_evidence(self):
        root = Path(__file__).resolve().parents[1]
        stored = json.loads((root / 'docs/evidence/recontact-self-imitation-v2/result.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            backend = case_tools.CaseTools.__new__(case_tools.CaseTools)
            backend.output = Path(temp)
            backend.physical = Path(temp) / 'physical'
            backend.conditions = {c['model']: c for c in stored['conditions']
                if c['seed'] == 64 and c['kind'] == 'full' and c['model'] in case_tools.MODELS}
            backend.verified_archive = {'passed': True}  # Test gate separately from boundary audit.
            backend.policy = case_tools.read(root / 'environment/agent-workflow-v1.json')['acceptance']
            backend.gate_hash = case_tools.sha(root / 'scripts/agent_workflow.py')
            backend.verification_attempted = False
            backend.new_conditions = None
            gate = backend.dispatch('check_acceptance', {'case': 'seed64', 'source': 'archived'})
            self.assertFalse(gate['accepted'])
            self.assertIn('per_seed_success_regression', gate['reasons'])
            self.assertIn('contact_force_regression', gate['reasons'])
            self.assertAlmostEqual(gate['relative_ceiling_N'], 262.29104052118595)
            self.assertTrue(gate['llm_cannot_override'])
            backend.verification_attempted = True
            with self.assertRaises(ValueError):
                backend.dispatch('check_acceptance', {'case': 'seed64', 'source': 'archived'})
            invalid = backend.dispatch('check_acceptance', {'case': 'seed64', 'source': 'new'})
            self.assertFalse(invalid['accepted'])
            self.assertIn('new_verification_incomplete_or_invalid', invalid['reasons'])


if __name__ == '__main__':
    unittest.main()
