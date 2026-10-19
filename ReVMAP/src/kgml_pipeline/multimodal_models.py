"""Compact multimodal baselines for drug-level MoA prediction.

The original DTA architectures predict one drug-target pair at a time. The
benchmark in this repository assigns one label to a drug, which can have many
targets. These adaptations therefore encode each target sequence and average
the target representations before classification.

``DruGNNosisStyleClassifier`` is a lightweight task-matched approximation of
the DruGNNosis-MoA design: it encodes drug/target biomedical text fields and
performs drug-target neighborhood aggregation before drug-level classification.
It is intentionally kept dependency-light rather than requiring a downloaded
SciBERT checkpoint, while preserving the same leakage-controlled outer-fold
protocol as the other baselines.
"""

from __future__ import annotations


def _torch_modules():
    import torch
    from torch import nn

    return torch, nn


class SequenceCNN:
    """Thin wrapper that delays importing torch until model construction."""

    def __new__(
        cls,
        vocab_size,
        embedding_dim=32,
        channels=64,
        kernel_sizes=(4, 8, 12),
        output_dim=128,
    ):
        torch, nn = _torch_modules()

        class _SequenceCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(
                    vocab_size,
                    embedding_dim,
                    padding_idx=0,
                )
                self.convolutions = nn.ModuleList([
                    nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size)
                    for kernel_size in kernel_sizes
                ])
                self.output = nn.Sequential(
                    nn.Linear(channels * len(kernel_sizes), output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )

            def forward(self, tokens):
                embedded = self.embedding(tokens).transpose(1, 2)
                pooled = []
                for convolution in self.convolutions:
                    values = torch.relu(convolution(embedded))
                    pooled.append(torch.amax(values, dim=-1))
                return self.output(torch.cat(pooled, dim=-1))

        return _SequenceCNN()


class MultiTargetMixin:
    def _encode_targets(self, target_tokens, target_mask):
        torch, _ = _torch_modules()
        batch_size, target_count, sequence_length = target_tokens.shape
        flat_tokens = target_tokens.reshape(batch_size * target_count, sequence_length)
        flat_mask = target_mask.reshape(batch_size * target_count)
        selected = flat_tokens[flat_mask]
        if selected.shape[0] == 0:
            return torch.zeros(
                batch_size,
                self.target_output_dim,
                device=target_tokens.device,
            )
        selected_features = self.target_encoder(selected)
        flat_features = torch.zeros(
            batch_size * target_count,
            selected_features.shape[-1],
            device=selected_features.device,
            dtype=selected_features.dtype,
        )
        flat_features[flat_mask] = selected_features
        features = flat_features.reshape(batch_size, target_count, -1)
        denominator = target_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return features.sum(dim=1) / denominator


class MultiTextTargetMixin:
    def _encode_target_texts(self, target_text_tokens, target_text_mask):
        torch, _ = _torch_modules()
        batch_size, target_count, sequence_length = target_text_tokens.shape
        flat_tokens = target_text_tokens.reshape(
            batch_size * target_count,
            sequence_length,
        )
        flat_mask = target_text_mask.reshape(batch_size * target_count)
        selected = flat_tokens[flat_mask]
        if selected.shape[0] == 0:
            return torch.zeros(
                batch_size,
                self.target_text_output_dim,
                device=target_text_tokens.device,
            )
        selected_features = self.target_text_encoder(selected)
        flat_features = torch.zeros(
            batch_size * target_count,
            selected_features.shape[-1],
            device=selected_features.device,
            dtype=selected_features.dtype,
        )
        flat_features[flat_mask] = selected_features
        features = flat_features.reshape(batch_size, target_count, -1)
        denominator = target_text_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return features.sum(dim=1) / denominator


class DeepDTAStyleClassifier:
    def __new__(
        cls,
        smiles_vocab_size,
        protein_vocab_size,
        class_count,
        hidden_dim=128,
    ):
        torch, nn = _torch_modules()

        class _DeepDTAStyle(nn.Module, MultiTargetMixin):
            def __init__(self):
                super().__init__()
                self.target_output_dim = hidden_dim
                self.smiles_encoder = SequenceCNN(
                    smiles_vocab_size,
                    output_dim=hidden_dim,
                )
                self.target_encoder = SequenceCNN(
                    protein_vocab_size,
                    kernel_sizes=(8, 12, 16),
                    output_dim=hidden_dim,
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, class_count),
                )

            def forward(
                self,
                smiles_tokens,
                target_tokens,
                target_mask,
                atom_features=None,
                normalized_adjacency=None,
                atom_mask=None,
                **unused,
            ):
                drug_features = self.smiles_encoder(smiles_tokens)
                target_features = self._encode_targets(target_tokens, target_mask)
                return self.classifier(torch.cat([drug_features, target_features], dim=-1))

        return _DeepDTAStyle()


class DenseGraphEncoder:
    def __new__(cls, atom_feature_dim, hidden_dim=128, layer_count=3):
        torch, nn = _torch_modules()

        class _DenseGraphEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Linear(atom_feature_dim, hidden_dim)
                self.layers = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(layer_count)
                ])
                self.dropout = nn.Dropout(0.15)

            def forward(self, atom_features, normalized_adjacency, atom_mask):
                values = torch.relu(self.input(atom_features))
                for layer in self.layers:
                    messages = torch.bmm(normalized_adjacency, values)
                    values = torch.relu(layer(messages)) + values
                    values = self.dropout(values)
                values = values * atom_mask.unsqueeze(-1)
                denominator = atom_mask.sum(dim=1, keepdim=True).clamp(min=1)
                return values.sum(dim=1) / denominator

        return _DenseGraphEncoder()


class GraphDTAStyleClassifier:
    def __new__(
        cls,
        atom_feature_dim,
        protein_vocab_size,
        class_count,
        hidden_dim=128,
    ):
        torch, nn = _torch_modules()

        class _GraphDTAStyle(nn.Module, MultiTargetMixin):
            def __init__(self):
                super().__init__()
                self.target_output_dim = hidden_dim
                self.graph_encoder = DenseGraphEncoder(
                    atom_feature_dim,
                    hidden_dim=hidden_dim,
                )
                self.target_encoder = SequenceCNN(
                    protein_vocab_size,
                    kernel_sizes=(8, 12, 16),
                    output_dim=hidden_dim,
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, class_count),
                )

            def forward(
                self,
                smiles_tokens,
                target_tokens,
                target_mask,
                atom_features,
                normalized_adjacency,
                atom_mask,
                **unused,
            ):
                drug_features = self.graph_encoder(
                    atom_features,
                    normalized_adjacency,
                    atom_mask,
                )
                target_features = self._encode_targets(target_tokens, target_mask)
                return self.classifier(torch.cat([drug_features, target_features], dim=-1))

        return _GraphDTAStyle()


class DruGNNosisStyleClassifier:
    def __new__(
        cls,
        text_vocab_size,
        class_count,
        hidden_dim=128,
        layer_count=2,
    ):
        torch, nn = _torch_modules()

        class _DruGNNosisStyle(nn.Module, MultiTextTargetMixin):
            def __init__(self):
                super().__init__()
                self.target_text_output_dim = hidden_dim
                self.drug_text_encoder = SequenceCNN(
                    text_vocab_size,
                    kernel_sizes=(3, 5, 7),
                    output_dim=hidden_dim,
                )
                self.target_text_encoder = SequenceCNN(
                    text_vocab_size,
                    kernel_sizes=(3, 5, 7),
                    output_dim=hidden_dim,
                )
                self.self_layers = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(layer_count)
                ])
                self.neighbor_layers = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(layer_count)
                ])
                self.dropout = nn.Dropout(0.25)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, class_count),
                )

            def forward(
                self,
                smiles_tokens=None,
                target_tokens=None,
                target_mask=None,
                atom_features=None,
                normalized_adjacency=None,
                atom_mask=None,
                drug_text_tokens=None,
                target_text_tokens=None,
                target_text_mask=None,
            ):
                if drug_text_tokens is None or target_text_tokens is None:
                    raise ValueError(
                        "DruGNNosisStyle requires drug and target text tokens"
                    )
                if target_text_mask is None:
                    target_text_mask = target_mask
                drug_text_features = self.drug_text_encoder(drug_text_tokens)
                target_text_features = self._encode_target_texts(
                    target_text_tokens,
                    target_text_mask,
                )
                graph_features = drug_text_features
                for self_layer, neighbor_layer in zip(
                    self.self_layers,
                    self.neighbor_layers,
                ):
                    update = torch.relu(
                        self_layer(graph_features)
                        + neighbor_layer(target_text_features)
                    )
                    graph_features = graph_features + self.dropout(update)
                return self.classifier(torch.cat([
                    drug_text_features,
                    target_text_features,
                    graph_features,
                ], dim=-1))

        return _DruGNNosisStyle()
