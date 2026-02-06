import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ClusterWarning, fcluster

from heatcluster.parse_files import parse_files

# Machine Learning Imports
try:
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except ImportError:
    silhouette_score = None
    PCA = None

from . import __version__

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%y-%b-%d %H:%M:%S",
    level=logging.INFO,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Genomic Distance Matrix")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input matrix file name"
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=[
            "snp-dists",
            "skani",
            "fastani",
            "mash",
            "sourmash",
            "melted",
            "poppunk",
            "ezaai",
            "nwk",
            "kwip",
            "ska",
            "bindash",
            "dashing",
            "pyani_identity",
            "pyani_errors",
            "iqtree",
            "gene_presenece_absense",
            "pathogen_detection",
            "vcf",
        ],
        default="snp-dists",
        help="Input file format.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="heatcluster_matrix.png",
        help="Output figure filename. Default: heatcluster_matrix.png",
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default="heatcluster_sorted.csv",
        help="Output sorted matrix CSV filename. Default: heatcluster_sorted.csv",
    )

    # ML / Clustering Extraction
    parser.add_argument(
        "-l",
        "--cluster-out",
        type=str,
        default="heatcluster_clusters.csv",
        help="Output filename for cluster assignments. Default: heatcluster_clusters.csv.",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        help="Split tree into K groups (e.g. 5). Overrides --cluster-t.",
    )
    parser.add_argument(
        "--cluster-t",
        type=float,
        help="Split tree by distance threshold (e.g. 10 SNPs or 0.05 ANI).",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically detect the optimal K using Silhouette Scores.",
    )

    # PCA
    parser.add_argument(
        "--pca", action="store_true", help="Generate a PCA scatter plot."
    )
    parser.add_argument(
        "--pca-out",
        type=str,
        default="heatcluster_pca.png",
        help="Filename for PCA plot.",
    )

    # Visual Appearance
    parser.add_argument(
        "--title", type=str, default="Heatmap", help="Title of the plot."
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Reds_r",
        help="Matplotlib colormap (e.g., Reds_r, viridis).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI (dots per inch) for output image. Default: 300.",
    )
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Do not show numbers inside the heatmap cells.",
    )

    # Dimensions & Text Size
    parser.add_argument("--width", type=float, help="Force figure width in inches.")
    parser.add_argument("--height", type=float, help="Force figure height in inches.")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale all font sizes by this factor.",
    )

    # Thresholds
    parser.add_argument("--vmin", type=float, help="Minimum value for color scale.")
    parser.add_argument("--vmax", type=float, help="Maximum value for color scale.")
    parser.add_argument("--hide-below", type=float, help="Mask values lower than this.")
    parser.add_argument(
        "--hide-above", type=float, help="Mask values higher than this."
    )

    parser.add_argument(
        "--no-cluster", action="store_true", help="Disable hierarchical clustering"
    )
    parser.add_argument(
        "--dendrogram",
        action="store_true",
        help="Show the dendrogram (tree) and borders",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def determine_figsize(
    num_samples: int,
    user_w: Optional[float] = None,
    user_h: Optional[float] = None,
    font_scale: float = 1.0,
) -> Tuple[float, float, Tuple[float, float]]:
    # Base Heuristics
    if num_samples >= 120:
        base_annot, base_label, base_size = 1.0, 2.0, (18, 15)
    elif num_samples >= 80:
        base_annot, base_label, base_size = 2.0, 3.0, (18, 15)
    elif num_samples >= 40:
        base_annot, base_label, base_size = 3.0, 4.0, (11, 8)
    elif num_samples >= 30:
        base_annot, base_label, base_size = 4.0, 6.0, (11, 8)
    else:
        base_annot, base_label, base_size = 6.0, 7.0, (11, 8)

    final_w = user_w if user_w else base_size[0]
    final_h = user_h if user_h else base_size[1]

    final_annot = round(base_annot * font_scale, 2)
    final_label = round(base_label * font_scale, 2)

    logging.info(f"Dimensions: {final_w}x{final_h} inches. Label Size: {final_label}.")
    return final_annot, final_label, (final_w, final_h)


def run_clustering(df: pd.DataFrame, args, font_tuple: tuple):
    matplotlib.use("agg")
    font_size, label_size, fig_size = font_tuple

    # Logic: Annotations
    if args.no_annot:
        annot_data = False
    else:
        annot_data = df.map(
            lambda x: "10K+" if x >= 10000 else str(x) if pd.notnull(x) else ""
        )

    # Logic: Masking
    mask = None
    if args.hide_below is not None or args.hide_above is not None:
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        if args.hide_below is not None:
            mask = mask | (df < args.hide_below)
        if args.hide_above is not None:
            mask = mask | (df > args.hide_above)

    logging.info(f"Generating heatmap: {args.output}")

    # Cluster assignments holder
    final_cluster_labels = None

    try:
        use_clustering = not args.no_cluster
        if use_clustering and df.shape[0] > 1 and df.shape[1] > 1:
            logging.info("Performing hierarchical clustering...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ClusterWarning)

                g = sns.clustermap(
                    df,
                    annot=annot_data,
                    fmt="",
                    cmap=args.cmap,
                    linewidths=0,
                    figsize=fig_size,
                    cbar_kws={"label": "Value"},
                    xticklabels=True,
                    yticklabels=True,
                    tree_kws={"linewidths": 1.5},
                    vmin=args.vmin,
                    vmax=args.vmax,
                    mask=mask,
                    annot_kws={"size": font_size},
                )

            if not args.dendrogram:
                g.ax_row_dendrogram.set_visible(False)
                g.ax_col_dendrogram.set_visible(False)
                g.ax_heatmap.set_frame_on(False)

            g.ax_heatmap.set_xticklabels(
                g.ax_heatmap.get_xmajorticklabels(),
                fontsize=label_size,
                rotation=45,
                ha="right",
            )
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_ymajorticklabels(), fontsize=label_size, rotation=0
            )
            g.fig.suptitle(args.title, fontsize=label_size + 4, y=1.02)

            plt.savefig(args.output, bbox_inches="tight", dpi=args.dpi)

            # --- ML FEATURE: CLUSTER EXTRACTION ---

            linkage_matrix = g.dendrogram_row.linkage

            # Logic: Do we need to calculate clusters?
            need_clusters = (
                args.cluster_out
                or args.auto_k
                or (args.pca and (args.cluster_k or args.cluster_t or args.auto_k))
            )

            if need_clusters:
                # OPTION 1: Auto-Detect K
                if args.auto_k:
                    if silhouette_score is None:
                        logging.error(
                            "scikit-learn not found. Install it to use --auto-k: pip install scikit-learn"
                        )
                        sys.exit(1)

                    logging.info("Running Silhouette Analysis to find optimal K...")
                    best_k = 2
                    best_score = -1
                    max_possible_k = min(10, len(df) - 1)

                    if max_possible_k < 2:
                        logging.warning(
                            "Not enough samples for Auto-K. Defaulting to K=2."
                        )
                    else:
                        # Prepare Distance Matrix
                        diag_mean = np.diag(df).mean()
                        if diag_mean > 90:
                            dist_matrix_np = (100 - df).to_numpy().copy()
                        elif diag_mean <= 1.0:
                            dist_matrix_np = (1.0 - df).to_numpy().copy()
                        else:
                            dist_matrix_np = df.to_numpy().copy()

                        np.fill_diagonal(dist_matrix_np, 0.0)
                        dist_matrix_np[dist_matrix_np < 0] = 0

                        for k in range(2, max_possible_k + 1):
                            labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
                            if len(set(labels)) < 2:
                                continue
                            score = silhouette_score(
                                dist_matrix_np, labels, metric="precomputed"
                            )
                            logging.info(f"  K={k}, Silhouette Score={score:.4f}")
                            if score > best_score:
                                best_score = score
                                best_k = k

                        logging.info(
                            f"Optimal K detected: {best_k} (Score: {best_score:.4f})"
                        )
                        args.cluster_k = best_k

                # OPTION 2: Extract Labels
                if args.cluster_k:
                    logging.info(f"Cutting tree into {args.cluster_k} clusters...")
                    final_cluster_labels = fcluster(
                        linkage_matrix, t=args.cluster_k, criterion="maxclust"
                    )
                elif args.cluster_t:
                    logging.info(f"Cutting tree at threshold {args.cluster_t}...")
                    final_cluster_labels = fcluster(
                        linkage_matrix, t=args.cluster_t, criterion="distance"
                    )
                elif (
                    args.cluster_out
                ):  # Fallback if writing file but no method provided
                    logging.warning("No clustering method provided. Defaulting to K=2.")
                    final_cluster_labels = fcluster(
                        linkage_matrix, t=2, criterion="maxclust"
                    )

                # Save to CSV if requested
                if args.cluster_out and final_cluster_labels is not None:
                    # Note: fcluster returns labels corresponding to original df index order
                    cluster_df = pd.DataFrame(
                        {"Sample": df.index, "Cluster_ID": final_cluster_labels}
                    )
                    cluster_df.to_csv(args.cluster_out, index=False)
                    logging.info(f"Saved cluster assignments to {args.cluster_out}")

            # Save Sorted Matrix
            reordered_index = g.dendrogram_row.reordered_ind
            sorted_df = df.iloc[reordered_index, reordered_index]
            sorted_df.to_csv(args.csv)
            logging.info(f"Saved sorted matrix to {args.csv}")

        else:
            logging.info("Using simple sorting (no clustering)...")
            df = df.loc[df.sum(axis=1).sort_values().index]
            if df.shape[0] == df.shape[1]:
                df = df.reindex(columns=df.index)

            fig, ax = plt.subplots(figsize=fig_size)
            sns.heatmap(
                df,
                annot=annot_data,
                fmt="",
                cbar_kws={"fraction": 0.01},
                cmap=args.cmap,
                linewidths=0,
                ax=ax,
                vmin=args.vmin,
                vmax=args.vmax,
                mask=mask,
                annot_kws={"size": font_size},
            )
            ax.set_title(args.title, fontsize=label_size + 4)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=label_size)
            plt.setp(
                ax.get_yticklabels(),
                rotation="horizontal",
                ha="right",
                fontsize=label_size,
            )
            plt.savefig(args.output, bbox_inches="tight", dpi=args.dpi)

            df.to_csv(args.csv)
            logging.info(f"Saved sorted matrix to {args.csv}")

        # --- ML FEATURE: PCA PLOT ---
        if args.pca:
            if PCA is None:
                logging.error("scikit-learn not found. Install it to use --pca.")
            else:
                logging.info("Generating PCA plot...")
                pca = PCA(n_components=2)

                # PCA on the distance matrix (PCoA equivalent)
                coords = pca.fit_transform(df)
                pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df.index)

                # Add Cluster Info if available
                if final_cluster_labels is not None:
                    pca_df["Cluster"] = final_cluster_labels.astype(str)
                    hue_col = "Cluster"
                else:
                    hue_col = None

                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    data=pca_df,
                    x="PC1",
                    y="PC2",
                    hue=hue_col,
                    palette="tab10",
                    s=100,
                    edgecolor="black",
                    alpha=0.8,
                )

                plt.title(f"PCA (PCoA) - {args.title}")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

                if hue_col:
                    plt.legend(
                        bbox_to_anchor=(1.05, 1), loc="upper left", title="Cluster"
                    )

                plt.savefig(args.pca_out, bbox_inches="tight", dpi=args.dpi)
                logging.info(f"Saved PCA plot to {args.pca_out}")

    except Exception as e:
        logging.error(f"Failed to create heatmap/PCA: {e}")
        sys.exit(1)
    finally:
        plt.close("all")


def main():
    parser = get_parser()
    args = parser.parse_args()

    try:
        plt.get_cmap(args.cmap)
    except ValueError:
        logging.error(f"Colormap '{args.cmap}' is not recognized.")
        sys.exit(1)

    input_file = Path(args.input)
    if not input_file.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    output_path = Path(args.output)
    valid_exts = [".png", ".pdf", ".svg", ".jpg", ".jpeg"]
    if output_path.suffix.lower() not in valid_exts:
        logging.error(f"Invalid output format '{output_path.suffix}'.")
        sys.exit(1)

    logging.info(f"Parsing and formatting {args.input}")
    df = parse_files(args.input, args.format)

    logging.info(f"Processing {len(df.columns)} samples")
    font_tuple = determine_figsize(
        len(df.columns), args.width, args.height, args.font_scale
    )

    run_clustering(df, args, font_tuple)
    logging.info("Done")


if __name__ == "__main__":
    main()
