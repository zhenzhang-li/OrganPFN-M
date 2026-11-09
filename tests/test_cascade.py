import numpy as np
import pandas as pd

from organpfnm.cascade import (
    AdmissionRule,
    admit_edges,
    benjamini_hochberg,
    maximum_strength_routes,
    temporal_disease_to_disease,
)


def test_bh_is_monotone_in_sorted_p_values():
    p = np.array([0.03, 0.001, np.nan, 0.02])
    q = benjamini_hochberg(p)
    order = np.argsort(p[np.isfinite(p)])
    assert np.all(np.diff(q[np.isfinite(q)][order]) >= -1e-12)
    assert np.isnan(q[2])


def test_temporal_disease_direction():
    data = pd.DataFrame({
        "onset_a": np.arange(150, dtype=float),
        "onset_b": np.arange(150, dtype=float) + 2.0,
    })
    edges = temporal_disease_to_disease(
        data, {"A": "onset_a", "B": "onset_b"},
        min_directional_pairs=100,
    )
    forward = edges.loc[(edges.source == "A") & (edges.target == "B")].iloc[0]
    assert forward["effect"] == 1.0
    assert bool(forward["eligible"])


def test_admission_uses_row_specific_negative_control_floor():
    edges = pd.DataFrame({
        "source": ["Body", "Brain"], "target": ["D", "D"],
        "effect": [1.2, 1.3], "p": [0.001, 0.001], "q": [0.01, 0.01],
        "count": [400, 400], "edge_type": ["O2D", "O2D"],
        "negative_control_floor": [1.1, 1.4],
    })
    result = admit_edges(edges, {"O2D": AdmissionRule("O2D", 300, 1.0)})
    assert result["source"].tolist() == ["Body"]


def test_maximum_product_route_prefers_stronger_path():
    edges = pd.DataFrame({
        "source": ["Body", "Body", "Mid"],
        "target": ["Disease", "Mid", "Disease"],
        "edge_type": ["O2D", "O2O", "O2D"],
        "strength": [0.5, 0.9, 0.9],
    })
    routes = maximum_strength_routes(edges, ["Body"], ["Disease"])
    assert routes.loc[0, "path"] == "Body -> Mid -> Disease"
    assert np.isclose(routes.loc[0, "joint_strength"], 0.81)
