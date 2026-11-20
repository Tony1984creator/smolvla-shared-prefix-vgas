"""Replay frozen shared-Bo4 query banks and compare critic weight formats."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.stats import spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method.precision import pack_weights, storage_bytes
from method.shared_prefix_selection import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "actor", "backbone", "critic", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    from safetensors.torch import load_model
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from smolvla_qchunk.eval.bestofn_eval import (
        _load_policy_config, _load_critic_checkpoint, BestOfNCriticTrainer,
        encode_policy_observations,
    )
    from models.smolvla.modeling_smolvla import SmolVLAPolicy

    args.output.mkdir(parents=True, exist_ok=False)
    groups = [[] for _ in range(10)]
    for path in sorted(args.archive.glob("*/B-Bo4_*/object_t*/query*/selection.json")):
        task = int(path.parents[1].name.split("_")[1][1:])
        groups[task].append(path)
    rng = random.Random(20261120)
    paths = [p for group in groups for p in sorted(rng.sample(group, 50))]
    ids = [str(p.relative_to(args.archive)) for p in paths]
    (args.output / "queries.json").write_text(json.dumps(ids, indent=2))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.mem_get_info()[0] < 4752 * 1024**2:
        raise RuntimeError("Insufficient GPU headroom")
    cfg = _load_policy_config(args.actor)
    cfg.vlm_model_name = str(args.backbone)
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    cfg.n_action_steps = 20
    policy = SmolVLAPolicy(cfg)
    policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    load_model(policy, str(args.actor / "model.safetensors"), strict=True)
    cfg.device = "cuda"
    policy.to("cuda").eval().requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(args.actor),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name}})
    checkpoint = _load_critic_checkpoint(args.critic)
    trainer = None
    rows = []
    for case_id, path in zip(ids, paths):
        archive = json.loads(path.read_text())
        instruction = json.loads((path.parents[1] / "result.json").read_text())["task_instruction"]
        with np.load(path.parent / "observation.npz") as obs:
            batch = preprocess_observation({"pixels": {k: obs[k][None].copy() for k in ("image","image2")},
                                            "agent_pos": obs["state"][None].copy()})
        batch["task"] = [instruction]
        batch = pre(batch)
        if trainer is None:
            checkpoint.config.action_samples = 4
            trainer = BestOfNCriticTrainer.build(policy, dict(batch, action=torch.zeros(1,32,7,device="cuda")),
                checkpoint.config, torch.device("cuda"),
                lambda p,b: encode_policy_observations(p,b,use_vlm_backbone_encode=True))
            trainer.load_state_dict(checkpoint.state_dict)
            trainer.target_critic = trainer.critic
            trainer.critic.eval().requires_grad_(False)
            models = {k: pack_weights(copy.deepcopy(trainer.critic), k)
                      for k in ("int8","float16","bfloat16")}
        values = {}
        def compare(module, inputs, kwargs):
            for label, model in models.items():
                heads = model(*inputs, **kwargs)
                values[label] = torch.stack(heads).min(0).values[:,0].tolist()
        handle = trainer.critic.register_forward_pre_hook(compare, with_kwargs=True)
        with torch.inference_mode():
            seed = archive["noise_seed"]
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn((16,50,cfg.max_action_dim),device="cuda",generator=generator)
            _, scores, _, _ = select(policy, trainer, batch, noise[:4])
        handle.remove()
        reference = scores.tolist()
        if not np.allclose(reference, archive["twin_min"], rtol=0, atol=0.001):
            raise ValueError(f"Archived score mismatch {case_id}: {reference}, {archive}")
        row = dict(query=case_id, original=reference, archived=archive["twin_min"], **values)
        rows.append(row)
    summary = {}
    for label in models:
        correlations = [float(spearmanr(r["original"], r[label]).statistic) for r in rows]
        summary[label] = dict(matches=sum(int(np.argmax(r[label]) == np.argmax(r["original"])) for r in rows),
                              mean_query_spearman=float(np.nanmean(correlations)),
                              defined_spearman=sum(int(np.isfinite(x)) for x in correlations),
                              bytes=storage_bytes(models[label]))
    result = dict(query_count=len(rows), original_bytes=storage_bytes(trainer.critic), formats=summary)
    (args.output / "scores.json").write_text(json.dumps(rows, indent=2))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
