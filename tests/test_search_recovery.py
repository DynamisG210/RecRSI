"""Offline fault injection at the native search and API boundaries."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.request import Request

import httpx
import pytest

NATIVE = Path(__file__).resolve().parents[1] / 'src/recrsi/native'
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Native runtime uses flock')


@pytest.fixture
def runtime(monkeypatch):
    for path in ('ledger', 'common/main_binding', 'common/comparison', 'task/search_source',
                 'task/search_source/ro', 'task/research_line',
                 'task/search_source/isolated_graph_core/src'):
        monkeypatch.syspath_prepend(str(NATIVE / path))


@pytest.fixture
def provider(runtime, tmp_path, monkeypatch):
    import main_relay_retry
    from main_api_ledger import MainApiLedger

    credentials = tmp_path / 'provider.conf'
    credentials.write_text('base_url = "https://example.invalid/v1"\napi_key = "fixture"\n')
    ledger = MainApiLedger(
        tmp_path / 'API_EVENTS.jsonl', 20, normalized_unit_usd=1, max_normalized_u=.11,
        model_prices={'gpt-5.6-terra': {'input': 2., 'output': 12., 'cache_read': 2., 'cache_creation': 2.}},
        max_input_tokens=128000, max_output_tokens=8000, input_estimate_margin=4096,
    )
    settings = dict(api_config=str(credentials), api_ledger=str(ledger.path),
                    resource_root=str(tmp_path), run_identity={'run': 'fixture'},
                    active_wall_path=str(tmp_path / 'ACTIVE_WALL.json'))
    from budget_binding import install_budget
    install_budget()
    from active_wall import ActiveWallClock
    ActiveWallClock.create(settings['active_wall_path'], settings['run_identity'])
    sent = []

    def respond(request):
        sent.append(request.content)
        return httpx.Response(200, json={
            'model': 'gpt-5.6-terra', 'status': 'completed',
            'output': [
                {'type': 'message', 'phase': 'commentary', 'content': [
                    {'type': 'output_text', 'text': 'commentary'}]},
                {'type': 'message', 'phase': 'final_answer', 'content': [
                    {'type': 'output_text', 'text': '{"mechanisms":["示例"]}'}]},
            ],
            'usage': {'input_tokens': 40, 'output_tokens': 12},
        })

    monkeypatch.setattr(main_relay_retry, 'async_client', lambda *args, **kwargs:
                        httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    return settings, ledger, sent


def test_autospace_resume_after_zero_send_and_after_paid_response(provider, monkeypatch, tmp_path):
    import graph_ro_generation as generation
    from resource_stop import SearchResourceStop
    from se_shared_api_ledger import BudgetExhausted

    settings, ledger, sent = provider
    monkeypatch.setattr(generation.method, 'validate', lambda value, stage: None)
    stage_root = tmp_path / 'autospace'
    request = {'instructions': 'fixture', 'fields': ['literal whitespace  preserved']}
    reserve = ledger.reserve

    def refuse(**kwargs):
        raise BudgetExhausted('fixture refused before dispatch')

    with monkeypatch.context() as m:
        m.setattr(ledger, 'reserve', refuse)
        with pytest.raises(SearchResourceStop, match='fixture refused'):
            generation.stage('GENESIS', request, stage_root, settings, ledger)
    assert not ledger.events() and not sent

    write_bytes = generation.write_bytes

    def interrupt_raw(path, data):
        if path.name == 'GENESIS_RAW_RESPONSE.json':
            raise OSError('fixture: controller stopped before publishing raw stage response')
        write_bytes(path, data)

    with monkeypatch.context() as m:
        m.setattr(generation, 'write_bytes', interrupt_raw)
        with pytest.raises(OSError, match='controller stopped'):
            generation.stage('GENESIS', request, stage_root, settings, ledger)
    assert len(sent) == 1
    assert json.loads(json.loads(sent[0])['input']) == request
    events = ledger.events()
    assert events[0]['estimated_input_tokens'] < 128000 - 4096
    assert events[0]['reserved_reference_usd'] < .11
    assert generation.stage('GENESIS', request, stage_root, settings, ledger) == {'mechanisms': ['示例']}
    assert len(sent) == 1
    assert ledger.events() == events

    reservation = events[0]
    (stage_root / 'GENESIS_RAW_RESPONSE.json').unlink()
    (tmp_path / 'durable_provider_responses' / (reservation['call_id'] + '.admitted')).unlink()
    with pytest.raises(RuntimeError, match='unsettled'):
        generation.stage('GENESIS', request, stage_root, settings, ledger)
    assert len(sent) == 1


def test_dispatch_identity_blocks_repeat_but_allows_distinct_calls(provider):
    from ro_main_provider import MainROTransport
    from resource_stop import SearchResourceStop
    from transport_durability import dispatch_key

    settings, ledger, sent = provider
    transport = MainROTransport(settings, ledger)
    request = Request('https://example.invalid/v1/responses', data=json.dumps({
        'model': 'gpt-5.6-terra', 'reasoning': {'effort': 'medium'},
        'input': 'fixture', 'max_output_tokens': 3000,
    }).encode())
    token = dispatch_key.set('sealed-logical-call-1')
    try:
        transport.urlopen(request)
        with pytest.raises(SearchResourceStop, match='requires reconciliation'):
            transport.urlopen(request)
        assert len(sent) == 1
        assert ledger.events()[0]['max_output_tokens'] == 3000
    finally:
        dispatch_key.reset(token)
    token = dispatch_key.set('sealed-logical-call-2')
    try:
        transport.urlopen(request)
    finally:
        dispatch_key.reset(token)
    assert sent == [request.data, request.data]


@pytest.mark.parametrize('schema_valid', [True, False])
def test_broker_recovers_paid_bytes_and_charges_original_role_budget(provider, monkeypatch, tmp_path, schema_valid):
    from recclaw_core.experiments.helix_abc_v1 import lab_api_broker as broker
    from recclaw_core.research_line import provider as research_provider
    from broker_failure import preserve_schema_failures
    import openevolve_provider
    import ro_main_provider
    import transport_durability
    monkeypatch.syspath_prepend(str(NATIVE / 'task/research_line/future_graph_composition'))
    from durable_broker_replay import durable_broker_scope

    settings, ledger, sent = provider
    monkeypatch.setattr(openevolve_provider, '_settings_and_ledger', lambda _: (settings, ledger))
    schema = tmp_path / 'schema.json'
    schema.write_text(json.dumps({
        'type': 'object', 'additionalProperties': False, 'required': ['proposals'],
        'properties': {'proposals': {'type': 'array', 'items': {'type': 'string'}}},
    }))
    # The shared transport's response fixture is schema-shaped for this test.
    import main_relay_retry
    content = '{"proposals":["fixture"]}' if schema_valid else '<think>fixture</think>{"proposals":[],"extra":1}'
    def respond(request):
        sent.append(request.content)
        return httpx.Response(200, json={'model': 'gpt-5.6-terra', 'status': 'completed',
            'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': content}]}],
            'usage': {'input_tokens': 40, 'output_tokens': 12}})
    monkeypatch.setattr(main_relay_retry, 'async_client', lambda *args, **kwargs:
                        httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    config_path = tmp_path / 'config.json'
    config_path.touch()
    kwargs = dict(call_root=tmp_path / 'broker', schema_path=schema, logical_call_id='selection',
                  session_id='fixture', prompt='choose', output_token_ceiling=100,
                  requested_model='gpt-5.6-terra')
    with ro_main_provider.native_ro_provider(config_path):
        with durable_broker_scope(broker, transport_durability, ledger.path), preserve_schema_failures():
            with ro_main_provider.native_ro_provider(config_path) as select:
                with monkeypatch.context() as m:
                    # Lose the controller after accounting, before SQLite's native receipt.
                    def stop_before_parse(*args, **kwargs):
                        raise KeyboardInterrupt
                    m.setattr(broker.LabApiCanaryBrokerV1, '_parse_response', stop_before_parse)
                    with pytest.raises(KeyboardInterrupt):
                        select(**kwargs)
                recovered = select(**kwargs)
                started, digest = research_provider._stored_provider_request_replay(
                    kwargs['call_root'], logical_call_id='selection', session_id='fixture',
                )
                assert started and digest == recovered.sealed_request['request_digest']
                cached = select(**kwargs, resume_request_digest=digest)
    for replay in (recovered, cached):
        if schema_valid:
            assert replay.call.response == {'proposals': ['fixture']}
        else:
            assert replay.call is None
            assert replay.failure['failure_class'] == 'SCHEMA_VALIDATION_FAILURE'
            monkeypatch.syspath_prepend(str(NATIVE / 'task/research/declaration'))
            from native_failure_bridge import NativeFailureBridge
            bridge = NativeFailureBridge(SimpleNamespace(settings=settings, ledger=ledger), frozen_session_id='fixture')
            evidence = tmp_path / 'durable_provider_responses' / ledger.events()[0]['call_id']
            decoded = bridge.resolve(replay, call_root=kwargs['call_root'],
                                     wire_bytes=evidence.with_suffix('.request').read_bytes(),
                                     response_bytes=evidence.with_suffix('.response').read_bytes(),
                                     require_proposal_count=False)
            assert decoded['original'] == {'proposals': [], 'extra': 1}
        assert replay.attempts[0]['physical_call_count'] == 0
        assert research_provider._physical_attempt_output_tokens(replay) == 12
    assert len(sent) == 1
    assert [event['event'] for event in ledger.events()] == ['RESERVED', 'SUCCESS']


def test_execution_error_pauses_without_sealing_result(runtime, tmp_path):
    from graph_native_launch import print_result, settle_result

    inputs = {'launch': {'output': str(tmp_path)}, 'hashes': {'fixture': 'sha'}}
    resources = SimpleNamespace(api=SimpleNamespace(summary=lambda: {'open_reservations': 0}))
    ledger = SimpleNamespace(snapshot=lambda: {'genuine_count': 2, 'winner': {'score': .3}})
    result = settle_result(inputs, {'status': 'EXECUTION_ERROR', 'reason': 'local crash'},
                           resources, ledger)
    assert not (tmp_path / 'RESULT.json').exists()
    assert json.loads((tmp_path / 'PAUSED.json').read_bytes())['genuine_count'] == 2
    with pytest.raises(SystemExit) as error:
        print_result(inputs, result)
    assert error.value.code == 1


@pytest.mark.parametrize('autospace', [True, False])
def test_confirmed_empty_relay_at_deadline_can_resume_same_stage(provider, monkeypatch, tmp_path, autospace):
    import time
    import graph_ro_generation as generation
    import main_relay_retry
    from resource_stop import SearchResourceStop
    from graph_native_launch import settle_result
    from ro_main_provider import MainROTransport
    from transport_durability import dispatch_key

    settings, ledger, sent = provider
    if autospace:
        monkeypatch.setattr(generation.method, 'validate', lambda value, stage: None)
        def call():
            return generation.stage('GENESIS', {}, tmp_path / 'autospace', settings, ledger)
    else:
        from auto_search_space import extract_responses_api_scientific_response
        transport = MainROTransport(settings, ledger)
        url = 'https://example.invalid/v1/responses'
        payload = {'model': 'gpt-5.6-terra', 'reasoning': {'effort': 'medium'},
                   'input': 'fixture', 'max_output_tokens': 3000}
        def call():
            token = dispatch_key.set('role')
            try:
                with transport.urlopen(Request(url, data=json.dumps(payload).encode())) as response:
                    return extract_responses_api_scientific_response(json.load(response))
            finally:
                dispatch_key.reset(token)

    def overloaded(request):
        sent.append(request.content)
        return httpx.Response(503, json={'error': {'message': 'servers overloaded'}})

    with monkeypatch.context() as m:
        m.setattr(main_relay_retry, 'async_client', lambda log_path, timeout, *, resources, call_id, **kwargs:
                  httpx.AsyncClient(transport=main_relay_retry.MainRelayTransport(
                      log_path, resources, call_id, inner=httpx.MockTransport(overloaded),
                      retry_deadline=time.monotonic())))
        with pytest.raises(SearchResourceStop) as error:
            call()
    assert error.value.retryable_interruption
    assert [event['event'] for event in ledger.events()] == ['RESERVED', 'EXTERNAL_EMPTY_STOP']
    inputs = {'launch': {'output': str(tmp_path)}, 'hashes': {'fixture': 'sha'}}
    resources = SimpleNamespace(api=ledger)
    evaluation = SimpleNamespace(snapshot=lambda: {'genuine_count': 0, 'winner': None})
    result = settle_result(inputs, {'status': 'RETRYABLE_INTERRUPTION'}, resources, evaluation)
    assert result['status'] == 'INTERRUPTED_RETRY_WITH_RESUME'
    assert not (tmp_path / 'PAUSED.json').exists()
    if not autospace:
        payload['input'] = 'different request'
        with pytest.raises(SearchResourceStop, match='requires reconciliation'):
            call()
        payload['input'] = 'fixture'
    assert call() == {'mechanisms': ['示例']}
    assert len(sent) == 2 and sent[0] == sent[1]


def test_task_facts_deduplication_requires_exact_values(runtime, monkeypatch):
    monkeypatch.syspath_prepend(str(NATIVE / 'task/research'))
    from run_graph_ro_main import common_prompt
    from source_bound_context import MARKER

    facts = {'training': {'epochs': 3}, 'task': 'literal  spacing'}
    context = {'objective': {'space': {'task': {**facts, 'dataset': 'fixture'}}}}
    prompt = MARKER + '\n' + json.dumps(context)
    assert 'See objective.space.task' in common_prompt(prompt, facts, {}, {})
    changed = {**facts, 'training': {'epochs': 4}}
    assert '"epochs":4' in common_prompt(prompt, changed, {}, {})
    assert '"task":"literal  spacing"' in common_prompt('selector prompt', facts, {}, {})


def test_gpu_admission_waits_for_the_configured_uuid(runtime, monkeypatch, tmp_path):
    import os
    from contextlib import contextmanager
    import physical_capacity

    source = tmp_path / 'candidate.py'
    source.write_text('pass\n')
    monkeypatch.setattr(physical_capacity, 'E', tmp_path.parent)
    cpus = sorted(os.sched_getaffinity(0))
    config = dict(output=str(tmp_path), cpus=cpus, gpu_uuid='GPU-assigned',
                  driver_root=str(tmp_path), resource_slots=str(tmp_path))
    free = iter(['GPU-assigned, 1024\n', 'GPU-assigned, 12000\n'])
    sleeps = []
    claims = []

    def query(command, **kwargs):
        assert command[1:3] == ['-i', config['gpu_uuid']]
        return next(free)

    @contextmanager
    def claim(slots, uuid, driver, size, owner):
        claims.append(uuid)
        yield 'fixture-lease'

    monkeypatch.setattr(physical_capacity.subprocess, 'check_output', query)
    monkeypatch.setattr(physical_capacity, 'gpu_capacity', claim)
    monkeypatch.setattr(physical_capacity.time, 'sleep', sleeps.append)
    with physical_capacity.physical_gpu(config, source, 'qualification') as placed:
        assert placed['gpu_uuid'] == config['gpu_uuid']
    assert len(sleeps) == 1 and claims == [config['gpu_uuid']]
    events = [json.loads(line) for line in (tmp_path / 'PHYSICAL_PLACEMENT.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events] == ['WAIT_SHARED_GPU_CAPACITY', 'CLAIM', 'RELEASE']


def test_settled_schema_failure_does_not_block_remaining_roles(runtime, monkeypatch, tmp_path):
    for path in ('task/research/mechanism_contrast', 'task/research/declaration',
                 'task/research/measured_execution'):
        monkeypatch.syspath_prepend(str(NATIVE / path))
    import fresh_schema_continuation
    import settled_schema_continuation_v2_1215 as settled
    import validation_boundary
    import campaign_declaration_consumer as consumer

    original = validation_boundary.ContractBoundary
    monkeypatch.setattr(validation_boundary, 'ContractBoundary', original)
    monkeypatch.setattr(consumer.CampaignDeclarationConsumer, 'install',
                        consumer.CampaignDeclarationConsumer.install)
    verified = []
    monkeypatch.setattr(settled, 'settled_trace', lambda *args: verified.append(args))
    fresh_schema_continuation.install()
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    state = campaign / 'CAMPAIGN_STATE.json'
    state.write_text('{"state":{"next_round_index":1}}')
    boundary = validation_boundary.ContractBoundary(tmp_path)
    boundary.record('provider_schema',
                    {'failure_class': 'SCHEMA_VALIDATION_FAILURE', 'logical_call_id': 'fixture-role'})
    boundary.require_new_request()
    assert not verified
    state.write_text('{"state":{"next_round_index":2}}')
    boundary.require_new_request()
    assert verified == [(tmp_path, 1, 'fixture-role')]
    boundary.record('qualification_metadata', {'reason': 'frozen source mismatch'})
    from resource_stop import SearchResourceStop
    with pytest.raises(SearchResourceStop, match='REGION_VALIDATION_CONTRACT_STOP') as error:
        boundary.require_new_request()
    assert error.value.requires_reconciliation
