import numpy as np
import pandas as pd

from organpfnm.pioaa import PIOAA, aggregate_age_gaps_by_participant


def test_pioaa_returns_panel_calibrated_scores():
    rng = np.random.default_rng(12)
    organs = ["Body", "Brain", "Renal"]
    common = rng.normal(size=(300, 1))
    panel = pd.DataFrame(common + rng.normal(scale=0.6, size=(300, 3)), columns=organs)
    panel.loc[240:, "Brain"] = np.nan
    healthy = np.zeros(300, dtype=bool)
    healthy[:220] = True
    model = PIOAA(organs, min_pairwise_n=20, min_calibration_n=20)
    scores = model.fit_transform(panel, healthy)
    assert list(scores.columns) == [
        "observed_organs", "total_deviation", "magnitude", "common_level",
        "asynchrony", "s_magnitude", "s_asynchrony",
    ]
    assert (scores["asynchrony"].dropna() >= -1e-10).all()
    assert scores.loc[:219, "s_asynchrony"].notna().all()


def test_visit_aggregation_averages_repeated_organs():
    data = pd.DataFrame({
        "Participant_ID": ["a", "a", "b"],
        "Body": [1.0, 3.0, 2.0],
        "Brain": [np.nan, 4.0, 5.0],
    })
    result = aggregate_age_gaps_by_participant(data, ["Body", "Brain"])
    assert result.loc[0, "Body"] == 2.0
    assert result.loc[0, "Brain"] == 4.0
