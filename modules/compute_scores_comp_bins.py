"""
Compute reconstruction distances and detection/attribution results.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score, roc_auc_score
from plots import normalize_column_of_nparrays
from tqdm import tqdm

def compute_scores_comp_bins(parquet_path, output_dir, real_dir, fake_dirs, complexity_bins):
    # Load data
    data_frame = pd.read_parquet(parquet_path)

    normalized_groups = []

    for cm, cm_group in data_frame.groupby("complexity_metric", sort=False, observed=True):
        cm_group = normalize_column_of_nparrays(cm_group, "complexity")
        normalized_groups.append(cm_group)

    data_frame = pd.concat(normalized_groups, ignore_index=True)

    detection_results = []

    # Loop over combinations of ["transform", "repo_id", "distance_metric"] and each complexity_metric
    combinations = data_frame.groupby(
        ["transform", "repo_id", "distance_metric", "complexity_metric"], sort=False, observed=True
    )

    for (transform, repo_id, dist_metric, complexity_metric), group_df in tqdm(combinations, desc="Processing combinations"):
        # Flatten complexity arrays for binning
        complexities = np.concatenate(group_df.complexity.values)

        # Bin the complexity values
        bins_edges = np.linspace(complexities.min(), complexities.max(), complexity_bins + 1)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        # Group by complexity bins
        for fake_dir in tqdm(fake_dirs, desc="Processing fake dirs", leave=False):
            # Filter real and fake scores
            real_mask = group_df["dir"] == real_dir.__str__()
            fake_mask = group_df["dir"] == fake_dir.__str__()

            for i in tqdm(range(complexity_bins), desc="Processing bins", leave=False):
                # Define the bin range
                bin_min, bin_max = bins_edges[i], bins_edges[i + 1]

                # Filter complexities and distances for real and fake data within the bin
                real_distances = []
                fake_distances = []

                for _, row in group_df[real_mask].iterrows():
                    # Get complexities and corresponding distances within the bin
                    indices = (np.array(row.complexity) >= bin_min) & (np.array(row.complexity) < bin_max)
                    real_distances.extend(np.array(row.distance)[indices])

                for _, row in group_df[fake_mask].iterrows():
                    # Get complexities and corresponding distances within the bin
                    indices = (np.array(row.complexity) >= bin_min) & (np.array(row.complexity) < bin_max)
                    fake_distances.extend(np.array(row.distance)[indices])

                if len(real_distances) > 0 and len(fake_distances) > 0:  # Ensure non-empty bins
                    # Combine scores and labels
                    y_score = np.concatenate([real_distances, fake_distances])
                    y_true = np.array([1] * len(real_distances) + [0] * len(fake_distances))

                    # Compute Scores
                    ap = average_precision_score(y_true=y_true, y_score=y_score)
                    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

                    # Save results
                    detection_results.append({
                        "fake_dir": str(fake_dir),
                        "transform": transform,
                        "repo_id": repo_id,
                        "distance_metric": dist_metric,
                        "complexity_metric": complexity_metric,
                        "complexity_bin": i,
                        "complexity_bin_center": bin_centers[i],
                        "ap": ap,
                        "auroc": auroc,
                        "num_real": len(real_distances),
                        "num_fake": len(fake_distances),
                    })

    # Convert detection results to DataFrame
    detection_results_df = pd.DataFrame(detection_results)

    # Adjust categorical columns
    categoricals = [
        "fake_dir",
        "transform",
        "repo_id",
        "distance_metric",
        "complexity_metric",
        "complexity_bin",
    ]

    # Convert relevant columns to categorical
    detection_results_df[categoricals] = detection_results_df[categoricals].astype("category")

    # Save to Parquet
    detection_results_df.to_parquet(output_dir / "combined_scores_comp_bins.parquet", index=False)


def main(args):
    output_dir = Path("output/compute_scores_comp_bins") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    compute_scores_comp_bins(
        args.combined_dist_comp_parquet,
        output_dir,
        args.real_dir,
        args.fake_dirs,
        args.complexity_bins
    )

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="default")
    parser.add_argument("--combined-dist-comp-parquet", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--complexity_bins", type=int, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
