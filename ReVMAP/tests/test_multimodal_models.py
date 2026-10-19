from __future__ import annotations

import unittest


class MultimodalModelShapeTest(unittest.TestCase):
    def test_deepdta_style_forward_shape(self):
        import torch

        from kgml_pipeline.multimodal_models import DeepDTAStyleClassifier

        model = DeepDTAStyleClassifier(40, 30, class_count=4, hidden_dim=16)
        output = model(
            torch.randint(0, 40, (3, 64)),
            torch.randint(0, 30, (3, 2, 64)),
            torch.tensor([[True, True], [True, False], [True, True]]),
            torch.zeros(3, 12, 8),
            torch.eye(12).repeat(3, 1, 1),
            torch.ones(3, 12),
        )
        self.assertEqual((3, 4), tuple(output.shape))

    def test_graphdta_style_forward_shape(self):
        import torch

        from kgml_pipeline.multimodal_models import GraphDTAStyleClassifier

        model = GraphDTAStyleClassifier(8, 30, class_count=4, hidden_dim=16)
        output = model(
            torch.randint(0, 40, (3, 64)),
            torch.randint(0, 30, (3, 2, 64)),
            torch.tensor([[True, True], [True, False], [True, True]]),
            torch.rand(3, 12, 8),
            torch.eye(12).repeat(3, 1, 1),
            torch.ones(3, 12),
        )
        self.assertEqual((3, 4), tuple(output.shape))


if __name__ == "__main__":
    unittest.main()
