from pathlib import Path

import pandas as pd
import mlflow


ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "processed" / "metadata"


def main() -> None:
    """Log the harmonized metadata files and summary stats to MLflow."""
    mlflow.set_experiment("multimodal_emotion_recognition")

    csv_files = sorted(META_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {META_DIR}")

    with mlflow.start_run(run_name="preprocessing_summary") as run:
        mlflow.log_param("label_mapping", "4_labels_angry_happy_sad_neutral")
        mlflow.log_param("metadata_dir", str(META_DIR))

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)

            mlflow.log_param(f"{csv_path.stem}_rows", len(df))
            mlflow.log_param(f"{csv_path.stem}_columns", ",".join(df.columns))

            label_counts = df["label"].value_counts().sort_index()
            for label, count in label_counts.items():
                mlflow.log_metric(f"{csv_path.stem}_{label}_count", int(count))

            mlflow.log_artifact(str(csv_path), artifact_path="metadata_csvs")

        print(f"MLflow run started: {run.info.run_id}")
        print("Open the MLflow UI with:")
        print("mlflow ui")


if __name__ == "__main__":
    main()
