"""Run the accepted P4 Object driver with only the critic weight representation changed."""
from pathlib import Path
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import numpy as np


def run_worker(root, manifest, destination):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from adapters.object_shared_selection import source
    code = source()
    changes = {
        "trainer.target_critic=trainer.critic":
            "from method.precision import pack_weights\n                        trainer.target_critic=pack_weights(trainer.critic, 'int8')",
        "ROOT/'outputs/p0_object_plus_20261112'/('libero_config_'+args.platform)":
            "out/'libero_config'",
    }
    for original, replacement in changes.items():
        if code.count(original) != 1:
            raise RuntimeError("P4 driver has changed")
        code = code.replace(original, replacement)
    driver = root / "adapters/object_shared_selection.py"
    sys.argv = [str(driver), "--actor","B","--condition","B-Bo4","--platform","clean",
                "--horizon","default","--output",str(destination),"--manifest",str(manifest)]
    exec(compile(code, str(driver), "exec"), {"__name__":"__main__", "__file__":str(driver)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root","offline","output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.manifest:
        run_worker(root, args.manifest, args.output)
        return
    check = json.loads(args.offline.read_text())
    if check["query_count"] != 500 or check["formats"]["int8"]["matches"] < 475:
        raise ValueError("Offline criterion was not met")
    archive = root / "outputs/p4_fixed_closed_loop_20261118"
    args.output.mkdir(parents=True, exist_ok=False)
    trials = [archive / "trial.json", *sorted((archive / "manifests").glob("*.json"))]
    frozen = json.loads((archive / "manifest.json").read_text())["cases"]
    scheduled = [c for p in trials for c in json.loads(p.read_text())["cases"]]
    if len(scheduled) != 300 or sorted(scheduled,key=lambda c:c["case_id"]) != sorted(frozen,key=lambda c:c["case_id"]):
        raise ValueError("Mismatch in frozen schedule")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    lanes = [gpu for gpu in devices for _ in range(2)]
    def run_batch(manifest, gpu):
        free = subprocess.check_output(["nvidia-smi","--id="+gpu,"--query-gpu=memory.free",
                                        "--format=csv,noheader,nounits"],text=True)
        if int(free) < 4752 * lanes.count(gpu):
            raise RuntimeError("Insufficient GPU headroom")
        dest = args.output / manifest.stem
        with (args.output / (manifest.stem+".log")).open("x") as log:
            subprocess.run(["timeout","-k","30s","40m",sys.executable,__file__,"--root",str(root),
                            "--offline",str(args.offline),"--output",str(dest),"--manifest",str(manifest)],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), check=True, stdout=log, stderr=subprocess.STDOUT)
        state = json.loads((dest/"state.json").read_text())
        if state["exit_code"] != 0:
            raise RuntimeError("Worker aborted")
        return state["completed"]
    def run_lane(index):
        return [r for manifest in trials[index::len(lanes)] for r in run_batch(manifest, lanes[index])]
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        rows = [r for lane in pool.map(run_lane, range(len(lanes))) for r in lane]
    final = {r["case_id"]:int(r["success"]) for r in rows}
    if len(rows) != 300 or set(final) != {c["case_id"] for c in frozen}:
        raise ValueError("Incomplete coverage")
    controls = json.loads((archive/"results.json").read_text())["rows"]
    paired = {}
    ids = [c["case_id"] for c in frozen]
    cases = {k: {"int8":v} for k,v in final.items()}
    for label in ("B-BC","B-Bo4"):
        base = {r["case_id"]: int(r["success"]) for r in controls[label]}
        for k in ids:
            cases[k][label] = base[k]
        delta = np.array([final[k] - base[k] for k in ids])
        draws = np.random.default_rng(42).choice(delta,(20000,300)).mean(1)*100
        paired[label] = dict(control_successes=sum(base.values()), delta_pp=100*delta.mean(),
                             ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                             rescues=int((delta>0).sum()), damages=int((delta<0).sum()))
    report = dict(episodes=300, successes=sum(final.values()), comparisons=paired,cases=cases)
    (args.output/"summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
