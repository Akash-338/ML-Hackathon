from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "gps_to_arc_distance_m",
    "gps_heading_difference_deg",
    "arc_length_m",
    "distance_from_arc_start_m",
    "fraction_along_arc",
    "time_delta_s",
    "gps_speed_mps",
    "candidate_rank",
)


class ClassifierScorer:
    """Load a saved pipeline and score candidate JSONL rows or DataFrames."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.model = joblib.load(self.artifact_dir / "model.joblib")
        self.metadata = __import__("json").loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.copy()
        feature_frame = _feature_frame(frame)
        probabilities = self.model.predict_proba(feature_frame)
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 in classes:
            positive_index = classes.index(1)
        else:
            positive_index = probabilities.shape[1] - 1
        return probabilities[:, positive_index]

    def predict_jsonl(self, rows: Iterable[Mapping[str, object]]) -> np.ndarray:
        return self.predict_frame(pd.DataFrame(rows))

    def score_by_observation(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["probability"] = self.predict_frame(frame)
        return (
            frame.sort_values(["trip_id", "observation_index", "probability"], ascending=[True, True, False])
            .groupby(["trip_id", "observation_index"], sort=False)
            .head(1)
            .copy()
        )


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "features" in frame:
        features = frame["features"].apply(
            lambda value: value if isinstance(value, Mapping) else {}
        )
        for name in FEATURE_NAMES[:-1]:
            if name not in frame:
                frame[name] = features.apply(lambda value: value.get(name))
    for name in FEATURE_NAMES:
        if name not in frame:
            frame[name] = np.nan
    return frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
