"""VGAS Object checkpoint loader for profile_decision.py (paths from environment)."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from models.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_model
from smolvla_qchunk.eval.bestofn_eval import (
    _load_policy_config, _load_critic_checkpoint, BestOfNCriticTrainer,
    encode_policy_observations,
)


def load():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    actor = Path(os.environ['ACTOR_DIR'])
    cfg = _load_policy_config(actor)
    cfg.vlm_model_name = os.environ['BACKBONE_DIR']
    cfg.load_vlm_weights = False
    cfg.device = 'cpu'
    cfg.n_action_steps = 20
    policy = SmolVLAPolicy(cfg)
    policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    load_model(policy, str(actor / 'model.safetensors'), strict=True)
    cfg.device = 'cuda'
    policy.cuda().eval().requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(actor),
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'tokenizer_processor': {'tokenizer_name': cfg.vlm_model_name}})
    with np.load(os.environ['OBS_NPZ']) as obs:
        batch = preprocess_observation({
            'pixels': {k: obs[k][None].copy() for k in ('image', 'image2')},
            'agent_pos': obs['state'][None].copy()})
    batch['task'] = [json.loads(Path(os.environ['EPISODE_JSON']).read_text())['task_instruction']]
    batch = pre(batch)
    ckpt = _load_critic_checkpoint(Path(os.environ['CRITIC_CKPT']))
    def encoder(p, b):
        return encode_policy_observations(p, b, use_vlm_backbone_encode=True)
    trainer = BestOfNCriticTrainer.build(policy, dict(batch, action=torch.zeros(1,32,7,device='cuda')),
                                          ckpt.config, torch.device('cuda'), encoder)
    trainer.load_state_dict(ckpt.state_dict)
    trainer.target_critic = trainer.critic
    trainer.critic.eval().requires_grad_(False)
    return policy, trainer, batch
