import numpy as np
from pathlib import Path

from chronograph.graphs.chunking import chunks
from chronograph.graphs.connectivity import pearson
from chronograph.graphs.graph_builder import build_file
from chronograph.graphs.node_features import band_power
from chronograph.graphs.sparsify import top_k


def test_pearson_and_top_k_have_fixed_graph_shape():
    rng = np.random.default_rng(42)
    window = rng.normal(size=(23, 1280)).astype(np.float32)
    graph = top_k(pearson(window), 5)
    assert graph.shape == (23, 23)
    assert np.count_nonzero(graph, axis=1).min() >= 5
    assert np.allclose(np.diag(graph), 0)
    assert np.all(graph >= 0)
    assert np.allclose(graph, graph.T)


def test_band_features_have_five_values_per_node():
    window = np.ones((23, 1280), dtype=np.float32)
    features = band_power(
        window, 256, {
            "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
            "beta": (13, 30), "gamma": (30, 40),
        }
    )
    assert features.shape == (23, 5)


def test_chunks_drop_boundary_crossing_windows():
    adjacency = np.zeros((61, 23, 23), dtype=np.float32)
    features = np.zeros((61, 23, 5), dtype=np.float32)
    labels = np.array(["ictal"] * 60 + ["preictal"])
    segment_ids = np.array(["seizure-1"] * 60 + ["seizure-2"])
    records = chunks(adjacency, features, labels, segment_ids, 60, 60)
    assert len(records) == 1
    assert records[0]["label"] == "ictal"


def test_graph_builder_skips_files_without_manifest_windows(tmp_path):
    source = tmp_path / "sample.npz"
    manifest = tmp_path / "segments.csv"
    np.savez(
        source,
        data=np.zeros((1, 23, 1280), dtype=np.float16),
        labels=np.array(["ictal"]),
        window_starts=np.array([0]),
    )
    manifest.write_text(
        "subject,file,start_seconds,end_seconds,label,seizure_id\n"
        "other,sample.edf,0,5,ictal,other:1\n"
    )
    assert build_file(
        source,
        tmp_path / "out.npz",
        "subject",
        manifest,
        256,
        {"delta": (0.5, 4.0)},
        1,
        12,
        12,
    ) == 0
