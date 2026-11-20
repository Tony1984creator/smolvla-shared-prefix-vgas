# Installation and evaluation

Use Linux x86-64, CPython 3.10.12, an NVIDIA GPU with a CUDA 12.6-compatible driver, and system EGL/OpenGL libraries. Run these commands from the repository root. The pinned Linux PyPI PyTorch 2.7.1 distribution reports `2.7.1+cu126` at runtime. LeRobot is 0.4.0, MuJoCo is 3.3.6 and robosuite is 1.4.1.

```bash
uv venv --python 3.10.12 .venv
uv pip install --python .venv/bin/python --no-deps -r requirements.txt
source .venv/bin/activate
mkdir -p vendor assets
git clone https://github.com/Jyugo-15/VGAS.git vendor/VGAS
git -C vendor/VGAS checkout --detach ef656cb
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git vendor/LIBERO
git -C vendor/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
hf download SemyonXu616/VGAS-5-shot --revision f23261d42938662851e438cee6e25b2ceaa05ceb \
  --include 'smolvla/5_SHOT/pretrained_model/*' 'object/vgas_critic/last.ckpt' 'long/vgas_critic/last.ckpt' \
  --local-dir assets/author_assets
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --include '*.json' '*.txt' '*.model' '*.jinja' --local-dir assets/backbone_metadata
python -c 'import torch; print(torch.__version__)'
```

The evaluator reads clean LIBERO directly from its checkout. The historical LIBERO-Plus reference is commit `4976dc30028e805ff8094b55501d532c48fec182`. Both reported suites run in clean LIBERO.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_libero.py --suite object --candidates 4 --impl shared --seed 42 --episodes 2
CUDA_VISIBLE_DEVICES=0 python scripts/eval_libero.py --suite object --candidates 4 --impl original --seed 42 --episodes 2
CUDA_VISIBLE_DEVICES=0 python tests/test_equivalence.py
```

Each JSON output line gives the suite, zero-based task and initial-state indices, seed, control steps, decisions and terminal success. Each additional episode advances the initial-state index and seed by one. Use the same indices and seed for paired runs. Select one candidate for BC. Use `--assets` for a separately stored checkpoint tree. The standalone examples evaluate the selected cases. The headline tables aggregate the published paired cohorts.
