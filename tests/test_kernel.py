import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch_gpu


if is_torch_available():
    import torch

    from models.kernel import liger_rotary
    from models.modeling_lckv import apply_rotary


@require_torch_gpu
class LigerRotaryTest(unittest.TestCase):
    def test_liger_rotary(self):
        # Test case: Test liger_rotary function
        q = torch.randn(2, 3, 4, 6, device="cuda")
        freq = torch.randn(1, 4, 3, device="cuda")
        embed = torch.cat((freq, freq), dim=-1)
        cos = embed.cos()
        sin = embed.sin()
        unsqueeze_dim = 1

        result_q = liger_rotary(q, cos, sin, unsqueeze_dim)
        self.assertEqual(result_q.shape, (2, 3, 4, 6))

        ref_q = apply_rotary(q, cos, sin, unsqueeze_dim)
        self.assertTrue(torch.allclose(result_q, ref_q))


if __name__ == '__main__':
    unittest.main()
