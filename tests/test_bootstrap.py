"""Public example to baseline receipt to prepared search, using real CPU training."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_demo_prepares_and_delivers_root(tmp_path):
    pytest.importorskip('torch')
    import recbole
    from recrsi.bootstrap import demo, worker_modules
    from recrsi.config import check_config, read
    from recrsi.prepare import prepare
    report = demo(tmp_path/'demo')
    assert report['device']=='cpu' and report['cost']['gpu_seconds']==0
    project=tmp_path/'demo/project'
    config=project/'task.json'
    env=read(project/'environment.json')
    api=tmp_path/'private.api-config'
    api.write_text('base_url = "https://example.invalid/v1"\napi_key = "fixture"\n')
    env.update(worker_python=sys.executable, recbole_root=str(Path(recbole.__file__).parent.parent),
               resource_slots=str(tmp_path), driver_root=str(tmp_path), api_config=str(api))
    (project/'environment.json').write_text(json.dumps(env))
    assert check_config(config)['status']=='INPUTS_VALIDATED_NO_EXECUTION'
    prepared=prepare(config,tmp_path/'prepared')
    task=read(prepared/'TASK.json')
    launch=read(task['launch'])
    evaluator=read(launch['evaluator_config'])
    # Exercise the runtime root consumer, not only the preparation validator.
    worker_modules()
    from budget_binding import install_budget
    install_budget()
    from graph_root_launcher import NoProviderRoot
    from active_wall import ActiveWallClock
    from main_resources import MainResources
    from graph_reference_root import reference_root
    ActiveWallClock.create(evaluator['active_wall_path'],evaluator['run_identity'])
    resources=MainResources(evaluator['output'],evaluator['run_identity'],evaluator['active_wall_path'],NoProviderRoot())
    baseline, ledger=reference_root(evaluator,resources)
    assert baseline['metrics']['ndcg_at_5']==report['metrics']['ndcg@5']
    assert reference_root(evaluator,resources)[0]==baseline
    inspected=subprocess.run([sys.executable,'-m','recrsi','inspect',str(prepared)],capture_output=True,text=True)
    assert inspected.returncode==0,inspected.stderr
    # Exercise cold resume through the prepared CLI with local HTTP. Interrupt
    # after the paid rejection reaches SQLite and before the round can settle.
    script = """
import json, sys
from pathlib import Path
import httpx
from recrsi.cli import main

prepared = Path(sys.argv[1])
resume = sys.argv[2] == 'resume'
sys.path.insert(0, str(prepared/'task/search_source/ro'))
sys.path.insert(0, str(prepared/'task/search_source/isolated_graph_core/src'))
import graph_ro_generation
from graph_autospace import normalized_space
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiCanaryBrokerV1
graph_ro_generation.generate = lambda inputs, **kwargs: normalized_space({
    'search_space_id': 'fixture-space', 'family': {'family_id': 'fixture-family'},
    'primitive_registry': {'axes': []}, 'grammar': {'operators': []}, 'capability_policy': {},
}, prepared)

async def post(self, url, *, content, **kwargs):
    payload = json.loads(content)
    assert 'OWN_COMPUTATION_CONTRASTS' in payload['input']
    assert 'EXECUTED_TASK_FACTS' in payload['input']
    assert 'common_protocol_sha256' not in payload['input']
    assert payload['max_output_tokens'] == 17000
    state = next(prepared.glob('task/research_line/runs/*/campaign/CAMPAIGN_STATE.json'))
    round_index = json.loads(state.read_bytes())['state']['next_round_index']
    if round_index >= 2:
        print('PREPARED_PRODUCER_REACHED_ROUND_TWO')
        raise SystemExit(24)
    assert not resume, 'Cold resume must use the paid round-one response'
    return httpx.Response(200, request=httpx.Request('POST', url), json={
        'model': payload['model'], 'status': 'completed',
        'output': [
            {'type': 'message', 'phase': 'commentary', 'content': [
                {'type': 'output_text', 'text': 'commentary'}]},
            {'type': 'message', 'phase': 'final_answer', 'content': [
                {'type': 'output_text', 'text': '{"proposals":[]}'}]},
        ],
        'usage': {'input_tokens': 50, 'output_tokens': 12},
    })

httpx.AsyncClient.post = post
if not resume:
    persist = LabApiCanaryBrokerV1._persist_failure
    def stop_after_paid_failure(self, **kwargs):
        persist(self, **kwargs)
        print('PREPARED_PAID_SCHEMA_REJECTION')
        raise SystemExit(23)
    LabApiCanaryBrokerV1._persist_failure = stop_after_paid_failure
sys.argv = ['recrsi', 'run', str(prepared)] + (['--resume'] if resume else [])
main()
"""
    first = subprocess.run([sys.executable,'-c',script,str(prepared),'start'],capture_output=True,text=True)
    assert first.returncode == 23 and 'PREPARED_PAID_SCHEMA_REJECTION' in first.stdout, first.stdout+first.stderr
    executed=subprocess.run([sys.executable,'-c',script,str(prepared),'resume'],capture_output=True,text=True)
    error_log = Path(evaluator['output'])/'EXECUTION_ERROR.log'
    detail = error_log.read_text() if error_log.exists() else ''
    pause = Path(evaluator['output'])/'PAUSED.json'
    if pause.exists():
        detail += json.dumps({'pause_reason': read(pause).get('reason')})
    assert executed.returncode == 24, executed.stdout+executed.stderr+detail
    assert 'PREPARED_PRODUCER_REACHED_ROUND_TWO' in executed.stdout, executed.stdout+executed.stderr+detail
