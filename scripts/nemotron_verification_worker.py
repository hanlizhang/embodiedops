"""Independent process adapter; executes only the frozen seed64 verification job."""
import argparse
from pathlib import Path
import sys

from nemotron_case_tools import ROOT, read, MODELS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    output=args.root.resolve()
    assert output.is_relative_to((ROOT/'outputs').resolve()) and output.name=='physical'
    assert args.job.resolve().parent==output
    plan=read(output/'registered-plan.json');job=read(args.job)
    assert plan['version']=='nemotron-agent-fixed-seed64-verification-v1'
    assert job in read(output/'jobs.json')
    assert job['model'] in MODELS and job['seed']==64 and job['kind']=='full' and job['cap']==600
    import recontact_v2
    import recontact_v2_worker
    # Reuse the exact existing worker implementation; redirect only its output root.
    recontact_v2.OUT=output
    recontact_v2_worker.OUT=output
    result=recontact_v2_worker.worker(job)
    return 0 if result['execution_validity'] else 1


if __name__=='__main__':
    sys.exit(main())
