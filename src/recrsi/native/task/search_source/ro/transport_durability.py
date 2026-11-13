"""Persist paid response bytes before SUCCESS accounting and broker handoff."""
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path

VERSION='graph-paid-response-durability-v1'
dispatch_key = ContextVar('ro_provider_dispatch_key', default=None)


def persist_paid_response(settings, call_id, label, payload, raw, request_data):
    root=Path(settings['api_ledger']).parent/'durable_provider_responses'
    root.mkdir(exist_ok=True)
    record=dict(version=VERSION,call_id=call_id,request_body_sha256=label,
        response_sha256=hashlib.sha256(raw).hexdigest(),model=payload['model'])
    # Store the exact wire request bytes through the caller, never credentials.
    assert hashlib.sha256(request_data).hexdigest()==label
    for suffix,data in (('.request',request_data),('.response',raw),('.json',json.dumps(record,sort_keys=True).encode())):
        target=root/(call_id+suffix)
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Conflicting durable response identity')
            continue
        write_bytes(target, data)


def write_bytes(path, data):
    """Publish complete bytes before another stage can consume the file."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
