from contextlib import contextmanager,ExitStack
from pathlib import Path
import json,os,sys,time,subprocess,hashlib
E=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(E/'root_runtime'))
from shared_gpu_capacity import gpu_capacity
@contextmanager
def physical_gpu(config,source,kind):
    output=Path(config['output']).resolve()
    assert E in output.parents
    assert set(os.sched_getaffinity(0))==set(config['cpus'])
    driver=config['driver_root'];slots=Path(config['resource_slots']);assigned=config['gpu_uuid']
    path=output/'PHYSICAL_PLACEMENT.jsonl';waiting=False
    def record(event,**kw):
        with path.open('a') as f:f.write(json.dumps(dict(event=event,time_ns=time.time_ns(),pid=os.getpid(),starttick=Path('/proc/self/stat').read_text().split()[21],kind=kind,source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),**kw))+'\n')
    while True:
        env=dict(os.environ,LD_LIBRARY_PATH=driver+':'+os.environ.get('LD_LIBRARY_PATH',''))
        rows=subprocess.check_output([driver+'/nvidia-smi','-i',assigned,
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],env=env,text=True)
        uuid, free = (field.strip() for field in rows.strip().split(','))
        if uuid != assigned:
            raise RuntimeError('GPU query did not return the assigned device')
        if int(free) >= 9472:
            with ExitStack() as stack:
                try:lease=stack.enter_context(gpu_capacity(slots,assigned,driver,9472,str(output)))
                except BlockingIOError:pass
                except RuntimeError as e:
                    if not str(e).startswith('Insufficient capacity:'):raise
                else:
                    record('CLAIM',gpu_uuid=assigned,cpus=config['cpus'],lease=lease)
                    try:yield config
                    finally:record('RELEASE',gpu_uuid=assigned)
                    return
        if not waiting:record('WAIT_SHARED_GPU_CAPACITY',gpu_uuid=assigned,gpu_reserved=False);waiting=True
        time.sleep(45)
