"""Single fresh Genesis/Critic, same MAIN ledger as native search, no redraw."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request
import graph_autospace as method
from transport_durability import write_bytes

def freeze(path,value):
    encoded=json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=encoded.encode():raise ValueError('Frozen graph input/output changed: '+str(path))
    else:write_bytes(path, encoded.encode())

def prepare(inputs):
    launch,config=inputs['launch'],inputs['evaluator']
    if config['phase']=='BOUNDED_GRAPH_DEVELOPMENT':
        if (launch['autospace_seed'],launch['search_seed'],config['train_seed'])!=(94001,43001,51001):
            raise ValueError('Graph development seeds differ from prospective binding')
    elif launch['autospace_seed']!=91025+launch['repeat']:
        raise ValueError('Formal AutoSpace seed differs from MAIN repeat')
    contract=method.build_contract(config,launch,inputs['facts'])
    request=method.genesis_request(contract,inputs['parent'],inputs['corpus'])
    root=Path(launch['output'])/'autospace'
    return root,contract,request

def stage(name,request,root,settings,ledger,*,transport=None):
    from recclaw_core.experiments.helix_abc_v1.provider_model_routing import require_returned_model_identity
    from auto_search_space import extract_responses_api_scientific_response
    from ro_main_provider import MainROTransport
    from se_shared_api_ledger import _load_route_credentials
    legacy_started = root/(name+'_STARTED.json')
    payload=dict(model='gpt-5.6-terra',reasoning={'effort':'medium'},store=False,
        max_output_tokens=ledger.max_output_tokens,text={'format':{'type':'json_object'}},input=method.canonical(request).decode())
    if legacy_started.exists():
        payload['input'] = json.dumps(request, sort_keys=True, ensure_ascii=False)
    freeze(root/(name+'_REQUEST.json'),request);freeze(root/(name+'_TRANSPORT.json'),payload)
    raw=root/(name+'_RAW_RESPONSE.json')
    if not raw.exists():
        if legacy_started.exists():
            raise RuntimeError('Original generation interruption needs reconciliation; no new draw')
        wire = json.dumps(payload, ensure_ascii=False).encode()
        channel = 'recclaw:autospace:' + name.lower()
        data = saved_stage_response(settings, ledger, channel, wire)
        if data is None:
            routefile=settings.get('model_api_configs',{}).get('gpt-5.6-terra',settings['api_config'])
            route=_load_route_credentials(routefile)
            http=Request(route['base_url'].rstrip('/')+'/responses',data=wire,
                headers={'Authorization':'Bearer '+route['api_key'],'Content-Type':'application/json'})
            with (transport or MainROTransport(settings, ledger, channel=channel)).urlopen(http,timeout=900) as response:
                data=response.read()
        write_bytes(raw, data)
    body=json.loads(raw.read_bytes())
    if body.get('status') != 'completed':
        raise ValueError('AutoSpace response did not complete')
    require_returned_model_identity(payload['model'], body.get('model'), wire_model=payload['model'])
    result=extract_responses_api_scientific_response(body)
    freeze(root/(name+'_RESPONSE.json'),result);method.validate(result,name.lower())
    return result


def saved_stage_response(settings, ledger, channel, wire):
    """The common reservation is the dispatch boundary for each AutoSpace stage."""
    events = ledger.events()
    empty = {event['call_id'] for event in events if event['event'] == 'EXTERNAL_EMPTY_STOP'}
    attempts = [event for event in events if event['event'] == 'RESERVED'
                and event['channel'] == channel and event['call_id'] not in empty]
    if not attempts:
        return None
    if len(attempts) != 1:
        raise RuntimeError('Multiple AutoSpace dispatches require reconciliation')
    reservation = attempts[0]
    call_id = reservation['call_id']
    label = hashlib.sha256(wire).hexdigest()
    successes = [event for event in events if event['call_id'] == call_id
                 and event['event'] == 'SUCCESS']
    saved = Path(settings['api_ledger']).parent/'durable_provider_responses'
    admission = saved/(call_id+'.admitted')
    if (reservation['label'] != label or len(successes) != 1
            or successes[0]['label'] != label or not successes[0]['usage_reported']
            or not admission.exists()):
        raise RuntimeError('Original AutoSpace dispatch is unsettled; reconcile before resume')
    data = (saved/(call_id+'.response')).read_bytes()
    if ((saved/(call_id+'.request')).read_bytes() != wire
            or admission.read_bytes() != hashlib.sha256(data).hexdigest().encode()):
        raise RuntimeError('Original AutoSpace paid response evidence differs')
    return data

def generate(inputs,*,stage_call=stage):
    from openevolve_provider import _settings_and_ledger
    root,contract,genesis=prepare(inputs)
    freeze(root/'GRAPH_AUTOSPACE_CONTRACT.json',contract)
    freeze(root/'INPUT_IDENTITY.json',dict(input_hashes=inputs['hashes'],method=method.METHOD))
    settings,ledger=_settings_and_ledger(inputs['evaluator']['provider_config'])
    draft=stage_call('GENESIS',genesis,root,settings,ledger)
    critic=stage_call('CRITIC',method.critic_request(genesis,draft),root,settings,ledger)
    assembled,quality=method.assemble(draft,critic,contract,inputs['parent'])
    freeze(root/'ASSEMBLED_SPACE.json',assembled);freeze(root/'QUALITY_REPORT.json',quality)
    return method.normalized_space(assembled,root)
