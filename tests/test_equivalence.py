"""Compare all released Object Bo4 candidates and critic scores on a real reset."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class Equivalence(unittest.TestCase):
    def test_object_bo4_fp32(self):
        from scripts.eval_libero import configure, critic, initial, load_actor, original, prepare
        with tempfile.TemporaryDirectory() as directory:
            configure(directory)
            from method.shared_prefix_selection import select
            assets = ROOT / "assets"
            tf32 = torch.backends.cuda.matmul.allow_tf32
            cudnn_tf32 = torch.backends.cudnn.allow_tf32
            self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", tf32)
            self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", cudnn_tf32)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            policy, pre, _ = load_actor(assets, fp32=True)
            env, obs = initial("object", 0, 0, 0)
            self.addCleanup(env.close)
            batch = prepare(pre, env, obs)
            trainer = critic(policy, batch, assets, "object", 4)
            trainer.target_critic.float()
            generator = torch.Generator(device="cuda").manual_seed(17)
            noise = torch.randn(4, policy.config.chunk_size, policy.config.max_action_dim,
                                generator=generator, device="cuda")
            with torch.inference_mode():
                pool, scores, best, _ = select(policy, trainer, batch, noise)
                reference, values, chosen = original(policy, trainer, batch, noise)
            torch.testing.assert_close(pool, reference, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(scores, values, atol=1e-4, rtol=1e-4)
            self.assertEqual(best, int(values.argmax()))
            torch.testing.assert_close(pool[best], chosen[0], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
