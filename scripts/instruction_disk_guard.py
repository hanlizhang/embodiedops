"""Read-only Windows storage telemetry; retain the original 2 GiB floor."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
FLOOR=2*1024**3

def sample(path):
    path=Path(path)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    ex=kernel.GetDiskFreeSpaceExW
    ex.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_ulonglong),ctypes.POINTER(ctypes.c_ulonglong),ctypes.POINTER(ctypes.c_ulonglong)]
    ex.restype=wintypes.BOOL
    available,total,free=ctypes.c_ulonglong(),ctypes.c_ulonglong(),ctypes.c_ulonglong()
    if not ex(str(path),ctypes.byref(available),ctypes.byref(total),ctypes.byref(free)):
        raise ctypes.WinError(ctypes.get_last_error())
    root=str(path.resolve().anchor)
    sectors,bytes_sector,free_clusters,total_clusters=[wintypes.DWORD() for _ in range(4)]
    legacy=kernel.GetDiskFreeSpaceW
    legacy.argtypes=[wintypes.LPCWSTR]+[ctypes.POINTER(wintypes.DWORD)]*4
    legacy.restype=wintypes.BOOL
    if not legacy(root,ctypes.byref(sectors),ctypes.byref(bytes_sector),ctypes.byref(free_clusters),ctypes.byref(total_clusters)):
        raise ctypes.WinError(ctypes.get_last_error())
    usage=shutil.disk_usage(path)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),path=str(path),resolved=str(path.resolve()),
        shutil_free=usage.free,win32_available_to_caller=available.value,win32_total_free=free.value,
        win32_total=total.value,cluster_free=free_clusters.value*sectors.value*bytes_sector.value,
        floor_bytes=FLOOR,above_floor=min(usage.free,available.value,free.value,free_clusters.value*sectors.value*bytes_sector.value)>FLOOR)

def snapshot():
    out=ROOT/'outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation'
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        paths={k:sample(p) for k,p in [('repository_C',ROOT),('junction',out),('physical_D',out.resolve()),('C_root','C:/')]},
        environment={k:os.environ.get(k) for k in ['TEMP','TMP','TMPDIR','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','NUMBA_CACHE_DIR','TORCH_HOME','LOCALAPPDATA']},
        tempfile_directory=tempfile.gettempdir())

def require_safe(record):
    assert all(r['above_floor'] for r in record['paths'].values()),'Disk reserve below original2GiB threshold'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label',required=True)
    a=ap.parse_args()
    folder=ROOT/'outputs/instruction-storage-guard-audit'
    folder.mkdir(exist_ok=True)
    data=snapshot()
    with (folder/f'{a.label}.json').open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    print(json.dumps(data,indent=2))

if __name__=='__main__':main()
