"""Validate the explicitly approved new-only D: junction; retain logical paths."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/agent-workflow-v1/instruction-control-v1/attempt02'

def evaluation_root():
    record=json.loads((STUDY/'evaluation-storage.json').read_text(encoding='utf-8-sig'))
    logical=STUDY/'evaluation'
    physical=Path('D:/EmbodiedOpsOutputs/instruction-control-v1-evaluation')
    assert record['verified'] and Path(record['logical'])==logical
    assert Path(record['physical'])==physical and logical.is_junction()
    assert logical.resolve()==physical.resolve()
    assert (logical/'storage-probe.txt').read_text(encoding='utf-8-sig').strip()==record['token']
    return logical

def episode_path(path):
    path=path.absolute()
    root=evaluation_root()
    assert path.parent==root and path.resolve().parent==root.resolve()
    return path
