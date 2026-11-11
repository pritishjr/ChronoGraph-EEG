import numpy as np
import torch

from chronograph.models.baselines.svm_rf import SVMBaseline
from chronograph.models.slate import SLATEModel
from chronograph.models.slate.spectral_encoding import SupraLaplacianEigenvectorEncoder


def test_model_a_svm_fit_and_predict():
    rng = np.random.default_rng(42)
    adjacency = rng.normal(size=(6, 3, 3)).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2])
    model = SVMBaseline()
    model.fit(adjacency, labels)
    assert model.predict(adjacency).shape == (6,)
    assert model.predict_proba(adjacency).shape == (6, 3)


def test_model_d_forward_and_backward():
    torch.manual_seed(42)
    adjacency = torch.randn(1, 2, 4, 4, requires_grad=True)
    node_features = torch.randn(1, 2, 4, 5, requires_grad=True)
    model = SLATEModel(node_features=5, hidden_dim=8, eigenvectors=2)
    logits, attention = model(adjacency, node_features)
    assert logits.shape == (1, 3)
    assert attention.shape == (1, 8, 8)
    logits.sum().backward()
    assert adjacency.grad is not None
    assert node_features.grad is not None


def test_model_d_eigenvectors_are_reused_from_cache(tmp_path):
    adjacency = torch.randn(1, 2, 4, 4)
    encoder = SupraLaplacianEigenvectorEncoder(2, tmp_path)
    first = encoder(adjacency)
    cached = list(tmp_path.glob("*.npy"))
    assert len(cached) == 1
    second = encoder(adjacency)
    torch.testing.assert_close(first, second)
