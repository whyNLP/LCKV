import unittest

from models.utils import LayerTypeParser
from transformers import is_torch_available
from transformers.testing_utils import require_flash_attn, require_torch_gpu


if is_torch_available():
    import torch

    from models.utils import flash_attention_forward
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


class LayerTypeParserTest(unittest.TestCase):
    def test_init_with_valid_input(self):
        # Test case: Initialization with valid input
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        self.assertEqual([item.attends_to for item in parser], [1, 2, 0])
        self.assertEqual([item.use_sliding_window for item in parser], [False, True, False])

    def test_init_with_invalid_input(self):
        # Test case: Initialization with invalid input
        with self.assertRaises(Exception):
            LayerTypeParser("invalid_input")

        with self.assertRaises(Exception):
            LayerTypeParser("0_1t_2")

    def test_init_with_empty_string(self):
        # Test case: Initialization with an empty string
        with self.assertRaises(Exception):
            LayerTypeParser("")

    def test_len(self):
        # Test case: Check the length of the LayerTypeParser object
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        self.assertEqual(len(parser), 3)

    def test_getitem(self):
        # Test case: Get the layer type information for a specific layer index
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        layer_type_info = parser[1]
        self.assertEqual(layer_type_info.attends_to, 2)
        self.assertEqual(layer_type_info.attends_top, True)
        self.assertEqual(layer_type_info.use_sliding_window, True)

    def test_use_sliding_window(self):
        # Test case: Check if there exists a layer that uses sliding window attention
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        self.assertEqual(parser.use_sliding_window(), True)

        layer_type = "0_0_2"
        parser = LayerTypeParser(layer_type)
        self.assertEqual(parser.use_sliding_window(), False)

    def test_attends_top(self):
        # Test case: Check if there exists a layer that attends to layers above it
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        self.assertEqual(parser.attends_top(), True)

        layer_type = "0s_0s_2"
        parser = LayerTypeParser(layer_type)
        self.assertEqual(parser.attends_top(), False)

    def test_iteration_plan(self):
        # Test case: Check the iteration plan for the layer types
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        iteration_plan = parser.iteration_plan(forward_passes=7, backward_passes=2)
        # Add assertions for the iteration plan
        self.assertEqual(len(iteration_plan), 9)

        # each layer should be updated exactly once
        updated_slices = [step.layer_slice for step in iteration_plan if step.update]
        updated_layers = [set(range(len(parser))[layer_slice]) for layer_slice in updated_slices]
        self.assertEqual(set.union(*updated_layers), set(range(len(parser))))

        # cyclic dependencies should be resolved
        self.assertEqual([step.requires_grad for step in iteration_plan], [False, False, False, False, False, False, False, True, True])


        # Test for the case where there is no cyclic dependency
        layer_type = "0_1s_2"
        parser = LayerTypeParser(layer_type)
        iteration_plan = parser.iteration_plan(forward_passes=7, backward_passes=2)
        self.assertEqual(len(iteration_plan), 1)
        self.assertTrue(iteration_plan[0].requires_grad)
        self.assertTrue(iteration_plan[0].update)

    def test_check(self):
        # Test case: Check if the layer type is valid
        num_hidden_layers = 3
        layer_type = "1_2s_0"
        parser = LayerTypeParser(layer_type)
        self.assertIsNone(parser.check(num_hidden_layers))

        # Test case: Check for invalid layer type
        num_hidden_layers = 3
        layer_type = "1_2s_3"
        parser = LayerTypeParser(layer_type)
        with self.assertRaises(Exception):
            parser.check(num_hidden_layers)

        num_hidden_layers = 3
        layer_type = "0_1_2s_0"
        parser = LayerTypeParser(layer_type)
        with self.assertRaises(Exception):
            parser.check(num_hidden_layers)


@require_torch_gpu
@require_flash_attn
class FlashAttentionForwardTest(unittest.TestCase):
    def test_no_diag(self):
        # Test case: Test flash_attention_forward with no_diag=True
        query_states = torch.randn(2, 5, 3, 4, dtype=torch.bfloat16, device="cuda")
        key_states = torch.randn(2, 6, 3, 4, dtype=torch.bfloat16, device="cuda")
        value_states = torch.randn(2, 6, 3, 4, dtype=torch.bfloat16, device="cuda")
        attention_mask = None
        query_length = 5
        is_causal = True
        no_diag = True

        result = flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
            no_diag=no_diag
        )

        self.assertEqual(result.shape, (2, 5, 3, 4))

        # Test case: attention_mask is not None, square attention matrix
        query_states = torch.randn(2, 6, 3, 4, dtype=torch.bfloat16, device="cuda")
        attention_mask = torch.ones(2, 6, dtype=torch.long, device="cuda")
        attention_mask[1, 2:] = 0
        result = flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
            no_diag=no_diag
        )

        self.assertEqual(result.shape, (2, 5, 3, 4))

    def test_with_diag(self):
        # Test case: Test flash_attention_forward with no_diag=False
        query_states = torch.randn(2, 5, 3, 4, dtype=torch.bfloat16, device="cuda")
        key_states = torch.randn(2, 6, 3, 4, dtype=torch.bfloat16, device="cuda")
        value_states = torch.randn(2, 6, 3, 4, dtype=torch.bfloat16, device="cuda")
        attention_mask = None
        query_length = 5
        is_causal = True
        no_diag = False

        result = flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
            no_diag=no_diag
        )

        ref = _flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
        )

        self.assertEqual(result.shape, (2, 5, 3, 4))
        self.assertTrue(torch.allclose(result, ref))


if __name__ == '__main__':
    unittest.main()
