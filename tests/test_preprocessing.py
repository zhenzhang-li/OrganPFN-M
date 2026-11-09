import json

import numpy as np
import pandas as pd

from organpfnm.preprocessing import (
    apply_missingness_threshold,
    assemble_visits_from_manifest,
    build_training_schema,
    harmonize_visit_rows,
    prepare_threshold_datasets,
)


def _table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    feature_map = {"Body": ["b1", "b2"], "Brain": ["r1", "r2"]}
    data = pd.DataFrame({
        "Participant_ID": ["p1", "p2", "p3"],
        "Sex": ["F", "M", "F"],
        "timepoint": ["T0", "T2", "T0"],
        "timepoint_id": [0, 1, 0],
        "Age": [50.0, 60.0, 70.0],
        "Disease": ["healthy", "disease", "healthy"],
        "is_healthy": [1, 0, 1],
        "b1": [1.0, 1.0, 1.0], "b2": [2.0, np.nan, np.nan],
        "r1": [3.0, 3.0, np.nan], "r2": [4.0, np.nan, np.nan],
    })
    return data, feature_map


def test_harmonization_keeps_visits_as_independent_rows():
    t0 = pd.DataFrame({"Participant_ID": [1], "Sex": ["F"], "Age": [50]})
    t2 = pd.DataFrame({"Participant_ID": [1], "Sex": ["F"], "Age": [55]})
    combined = harmonize_visit_rows(t0, t2)
    assert len(combined) == 2
    assert combined["Participant_ID"].tolist() == [1, 1]
    assert combined["timepoint"].tolist() == ["T0", "T2"]


def test_threshold_and_minimum_two_organs():
    data, feature_map = _table()
    strict, _ = apply_missingness_threshold(data, feature_map, 0, min_organs=2)
    assert strict["Participant_ID"].tolist() == ["p1"]
    permissive, _ = apply_missingness_threshold(data, feature_map, 50, min_organs=2)
    assert permissive["Participant_ID"].tolist() == ["p1", "p2"]


def test_output_has_one_sex_column_and_internal_schema_slots(tmp_path):
    data, feature_map = _table()
    prepare_threshold_datasets(data, feature_map, tmp_path, thresholds=[50], min_organs=2)
    output = pd.read_csv(tmp_path / "thr_50" / "ssl" / "ssl_subset.csv")
    assert [column for column in output if column == "Sex"] == ["Sex"]
    assert not any(column.startswith("__model_") for column in output)
    schema = json.loads((tmp_path / "thr_50" / "ssl" / "schema.json").read_text(encoding="utf-8"))
    assert schema["n_measurement_features"] == 4
    assert schema["n_total_features"] == 6
    assert schema["filter_thresholds"]["ssl"]["min_organs"] == 2


def test_schema_separates_data_and_loader_thresholds():
    schema = build_training_schema({"A": ["a"], "B": ["b"]}, threshold=90)
    assert schema["filter_thresholds"]["ssl"]["max_missing_fraction"] == 0.9
    assert schema["filter_thresholds"]["ssl"]["organ_presence_threshold"] == 0.3
    assert schema["filter_thresholds"]["finetune"]["organ_presence_threshold"] == 0.5


def test_manifest_assembles_organ_files_as_separate_visits(tmp_path):
    for visit, age in (("T0", 50), ("T2", 55)):
        pd.DataFrame({
            "source_id": [1], "source_sex": ["F"], "source_age": [age],
            "source_disease": [""], "source_healthy": [1],
        }).to_csv(tmp_path / f"{visit}_meta.csv", index=False)
        pd.DataFrame({"source_id": [1], f"{visit}_value": [age / 10]}).to_csv(
            tmp_path / f"{visit}_organ.csv", index=False
        )
    manifest = {
        "participant_column": "Participant_ID",
        "visits": {
            visit: {
                "timepoint_id": index,
                "metadata": {
                    "path": f"{visit}_meta.csv",
                    "columns": {
                        "source_id": "Participant_ID", "source_sex": "Sex",
                        "source_age": "Age", "source_disease": "Disease",
                        "source_healthy": "is_healthy",
                    },
                },
                "organ_tables": {
                    "Body": {
                        "path": f"{visit}_organ.csv",
                        "columns": {"source_id": "Participant_ID"},
                        "keep": [f"{visit}_value"],
                    }
                },
            }
            for index, visit in enumerate(("T0", "T2"))
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = assemble_visits_from_manifest(path)
    assert result["Participant_ID"].tolist() == [1, 1]
    assert result["timepoint"].tolist() == ["T0", "T2"]
