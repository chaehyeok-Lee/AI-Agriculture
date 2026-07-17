"""Train four curtain-derived modules and create soil_temp-only test_y.csv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib-cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from soil_temp import SoilTemperaturePipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("input/dataset"))
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_continuous_outputs(pipe, detail, output_dir: Path):
    """Save observed and predicted curves on the anonymous inferred sequence."""
    continuous_dir = output_dir.parent / "completed_soil_temperature_png"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    for chunk in range(1, 5):
        parts = []
        train_dats = pipe.train_chunks.loc[
            pipe.train_chunks["chunk"].eq(chunk), "dat"
        ].tolist()
        for dat in train_dats:
            sequence_index = pipe.sequence_positions[dat]
            day = pipe.train_frame.loc[pipe.train_frame["dat"].eq(dat)].sort_values("minute")
            parts.append(pd.DataFrame({
                "time_within_dat": [value[-5:] for value in day["time"]],
                "soil_temp": day["soil_temp"].to_numpy(float),
                "value_type": "observed", "dat": dat,
                "sequence_index": sequence_index,
            }))
        test_dats = detail.loc[detail["chunk"].eq(chunk), "dat"].drop_duplicates().tolist()
        for dat in test_dats:
            sequence_index = pipe.sequence_positions[dat]
            day = detail.loc[detail["dat"].eq(dat)]
            parts.append(pd.DataFrame({
                "time_within_dat": [value[-5:] for value in day["time"]],
                "soil_temp": day["soil_temp"].to_numpy(float),
                "value_type": "predicted", "dat": dat,
                "sequence_index": sequence_index,
            }))
        continuous = pd.concat(parts, ignore_index=True).sort_values(
            ["sequence_index", "time_within_dat"]
        ).reset_index(drop=True)
        csv_path = continuous_dir / f"chunk{chunk}_continuous_soil_temperature.csv"
        continuous.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(16, 5))
        positions = sorted(continuous["sequence_index"].unique())
        segments, current = [], [positions[0]]
        for position in positions[1:]:
            if position != current[-1] + 1:
                segments.append(current)
                current = [position]
            else:
                current.append(position)
        segments.append(current)
        for segment in segments:
            group = continuous.loc[continuous["sequence_index"].isin(segment)]
            hhmm = group["time_within_dat"].str.split(":", expand=True).astype(int)
            x = group["sequence_index"].to_numpy(float) + (
                hhmm[0].to_numpy() * 60 + hhmm[1].to_numpy()
            ) / 1440.0
            ax.plot(x, group["soil_temp"], linewidth=1.25, color="#1565c0")
        ax.set_title(f"Chunk {chunk} soil temperature by inferred sequence")
        ax.set_xlabel("Anonymous sequence index")
        ax.set_ylabel("soil_temp")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            continuous_dir / f"chunk{chunk}_continuous_soil_temperature.png",
            dpi=150,
        )
        plt.close(fig)


def main():
    args = parse_args()
    train_x = pd.read_csv(args.input / "train/env/train_X.csv")
    train_y = pd.read_csv(args.input / "train/env/train_y.csv")
    test_x = pd.read_csv(args.input / "test/env/test_X.csv")
    pipe = SoilTemperaturePipeline(seed=args.seed).fit(
        train_x, train_y, reference_x=test_x
    )
    cv_scores = pipe.cross_validate(n_splits=4)
    forward_scores = pipe.forward_validate()
    prediction, detail = pipe.predict(test_x)
    args.output.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output / "test_y.csv", index=False)
    prediction.to_csv(args.output / "soil_temp_full_predictions.csv", index=False)
    detail.to_csv(args.output / "soil_temp_predictions_by_chunk.csv", index=False)
    if pipe.oof_predictions is None:
        raise RuntimeError("cross-validation did not create OOF predictions")
    pipe.oof_predictions.to_csv(
        args.output / "soil_temp_oof_predictions.csv", index=False
    )
    pipe.train_chunks.to_csv(args.output / "train_chunk_assignment.csv", index=False)
    with open(args.output / "cv_scores.json", "w", encoding="utf-8") as f:
        json.dump(cv_scores, f, ensure_ascii=False, indent=2)
    with open(
        args.output / "forward_cv_scores.json", "w", encoding="utf-8"
    ) as f:
        json.dump(forward_scores, f, ensure_ascii=False, indent=2)
    joblib.dump(pipe, args.output / "soil_temp_models.joblib")
    save_continuous_outputs(pipe, detail, args.output)
    for chunk, group in detail.groupby("chunk", sort=True):
        fig, ax = plt.subplots(figsize=(13, 4.5))
        for dat, day in group.groupby("dat", sort=False):
            x = day["time"].str[-5:]
            ax.plot(x, day["soil_temp"], linewidth=1.4, label=dat)
        ax.set_title(f"Chunk {chunk} predicted soil temperature (DAT only)")
        ax.set_xlabel("time within DAT")
        ax.set_ylabel("soil_temp")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        ticks = list(range(0, 288, 36))
        ax.set_xticks(ticks, [group.iloc[i]["time"][-5:] for i in ticks])
        fig.tight_layout()
        fig.savefig(args.output / f"chunk{chunk}_soil_temp_prediction.png", dpi=150)
        plt.close(fig)
    fold_headers = [f"Fold {i}" for i in range(1, len(cv_scores["per_fold_rmse"]) + 1)]
    headers = ["모델", *fold_headers, "평균 RMSE"]
    values = [
        "soil_temp",
        *(f"{score:.6f}" for score in cv_scores["per_fold_rmse"]),
        f"{cv_scores['fold_rmse_mean']:.6f}",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    print("| " + " | ".join(values) + " |")
    print(
        f"saved {len(pipe.oof_predictions):,} OOF rows -> "
        f"{args.output / 'soil_temp_oof_predictions.csv'}"
    )
    print(
        f"forward pooled RMSE: {forward_scores['pooled_rmse']:.6f} -> "
        f"{args.output / 'forward_cv_scores.json'}"
    )
    print(f"saved {len(prediction):,} rows -> {args.output / 'test_y.csv'}")


if __name__ == "__main__":
    main()
