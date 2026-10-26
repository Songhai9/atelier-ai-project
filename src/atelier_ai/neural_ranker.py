"""Shared neural ranking architecture and identifier encoding."""

import torch
from torch import nn


class NeuralRanker(nn.Module):
    """Score candidates using identifier embeddings and numeric features."""

    def __init__(
        self,
        n_customers,
        n_articles,
        n_numeric,
        embedding_dim=32,
        hidden_dims=(128, 64),
        dropout=0.2,
    ):
        super().__init__()

        self.customer_embedding = nn.Embedding(
            n_customers, embedding_dim, padding_idx=0
        )
        self.article_embedding = nn.Embedding(
            n_articles, embedding_dim, padding_idx=0
        )

        input_dim = n_numeric + 2 * embedding_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, numeric, customer, article):
        customer_vectors = self.customer_embedding(customer)
        article_vectors = self.article_embedding(article)

        combined = torch.cat(
            [numeric, customer_vectors, article_vectors],
            dim=-1,
        )
        return self.head(combined).squeeze(-1)


def encode_ids(values, vocabulary):
    """Map unseen identifiers to the reserved index zero."""
    return (
        values.map(vocabulary)
        .fillna(0)
        .to_numpy(dtype="int64", copy=True)
    )
