"""Device selection and tiny-model CPU versus XPU parity checks.

Run with ``python -m pytest -q tests`` from the repository root.
"""

import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.modeling_spirit_vla import BaseDiT, BaseDiTConfig  # noqa: E402
from robochallenge import run_robochallenge  # noqa: E402
from robochallenge.runner import executor as executor_mod  # noqa: E402
from utils import device as device_mod  # noqa: E402

HAS_XPU = device_mod.is_available("xpu")


def _fake_host(xpu=False, cuda=False):
    return mock.patch.multiple(
        device_mod,
        is_available=lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu"),
    )


class ResolveDeviceTest(unittest.TestCase):
    def test_cpu_only_host(self):
        with _fake_host():
            self.assertEqual(device_mod.resolve_device(), torch.device("cpu"))
            self.assertEqual(device_mod.resolve_device("auto"), torch.device("cpu"))

    def test_cuda_host(self):
        with _fake_host(cuda=True):
            self.assertEqual(device_mod.resolve_device(), torch.device("cuda", 0))

    def test_xpu_host_prefers_xpu(self):
        with _fake_host(xpu=True, cuda=True):
            self.assertEqual(device_mod.resolve_device(), torch.device("xpu", 0))

    def test_explicit_device_wins(self):
        with _fake_host(xpu=True):
            self.assertEqual(device_mod.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(device_mod.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(device_mod.resolve_device(torch.device("xpu", 1)), torch.device("xpu", 1))

    def test_bare_accelerator_gets_index_zero(self):
        self.assertEqual(device_mod.resolve_device("xpu"), torch.device("xpu", 0))
        self.assertEqual(device_mod.resolve_device("cuda"), torch.device("cuda", 0))


class ExecutorDispatchTest(unittest.TestCase):
    def test_device_flag_defaults_to_autodetect(self):
        args = run_robochallenge.build_parser().parse_args(
            ["--single_task", "t", "--robochallenge_job_id", "j", "--ckpt_path", "c", "--user_token", "u"]
        )
        self.assertIsNone(args.device)

    def _build(self, host, device, tmp):
        cfg = mock.Mock(
            single_task="move_objects_into_box",
            robochallenge_job_id="job",
            used_chunk_size=60,
            ckpt_path="ckpt",
            device=device,
            raw_embodiment_stats_json_path=None,
            use_embodiment_specific_norm=False,
        )
        policy = mock.MagicMock()
        with host, mock.patch.object(
            executor_mod.SpiritVLAPolicy, "from_pretrained", return_value=policy
        ) as loader, contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.chdir(tmp))
            executor = executor_mod.RoboChallengeExecutor(cfg)
        loader.assert_called_once_with("ckpt")
        return executor, policy

    def test_executor_moves_policy_to_detected_device(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            executor, policy = self._build(_fake_host(xpu=True), None, tmp)
        self.assertEqual(executor.device, torch.device("xpu", 0))
        policy.to.assert_called_once_with(torch.device("xpu", 0))

    def test_executor_explicit_device_wins(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            executor, policy = self._build(_fake_host(xpu=True), "cpu", tmp)
        self.assertEqual(executor.device, torch.device("cpu"))
        policy.to.assert_called_once_with(torch.device("cpu"))


def _tiny_dit():
    config = BaseDiTConfig(
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=2,
        interleave_self_attention=True,
        cross_attention_dim=24,
    )
    model = BaseDiT(**config.param_dict)
    generator = torch.Generator().manual_seed(0)
    # Re-randomise every parameter so that zero-initialised layers cannot hide a mismatch.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.2)
    return model.eval()


@unittest.skipUnless(HAS_XPU, "needs an Intel XPU")
class TinyParityTest(unittest.TestCase):
    def test_dit_matches_cpu(self):
        model = _tiny_dit()
        generator = torch.Generator().manual_seed(1)
        hidden = torch.randn(1, 6, 16, generator=generator)
        encoder = torch.randn(1, 5, 24, generator=generator)
        timestep = torch.tensor([0.5])
        with torch.no_grad():
            ref = model(hidden, encoder, timestep)
            xpu = model.to("xpu")(hidden.to("xpu"), encoder.to("xpu"), timestep.to("xpu")).cpu()
        self.assertGreater(ref.std().item(), 1e-3)
        torch.testing.assert_close(xpu, ref, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
