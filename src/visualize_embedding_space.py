"""Visualise the embedding space used by the KNN probe."""

import argparse

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from umap import UMAP  # noqa: E402

from src.config import ModelConfig, get_model_config  # noqa: E402
from src.data import load_label_map, load_splits  # noqa: E402
from src.runtime import SEED  # noqa: E402

pio.renderers.default = "browser"

_TAB10_HEX = [mcolors.to_hex(plt.get_cmap("tab10")(i)) for i in range(10)]


def _snippet(text, max_len: int = 180) -> str:
    return str(text)[:max_len].replace("\n", " ")


def load_knn_artifacts(model_cfg: ModelConfig, task: str, dataset_variant: str) -> dict:
    """Load the embeddings, labels and texts the plots need."""
    results_dir = model_cfg.results_dir(task, dataset_variant)
    clf = joblib.load(results_dir / "checkpoint" / "knn_classifier.joblib")

    train_embeddings = np.asarray(clf._fit_X)
    train_label_ids = np.asarray(clf._y)
    test_embeddings = np.load(results_dir / "cls_embeddings.npy")

    df = load_splits(dataset_variant)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if len(train_df) != len(train_embeddings) or not np.array_equal(
        train_df["label_id"].to_numpy(), train_label_ids
    ):
        raise ValueError(
            f"{results_dir}: train split ({len(train_df)} rows) doesn't line up with the "
            f"fitted KNN's embeddings ({len(train_embeddings)} rows) -- re-run the "
            f"probe's fit or check dataset_variant."
        )
    if len(test_df) != len(test_embeddings):
        raise ValueError(
            f"{results_dir}: test split ({len(test_df)} rows) doesn't line up with "
            f"cls_embeddings.npy ({len(test_embeddings)} rows)."
        )

    labels = load_label_map()["labels"]
    return {
        "results_dir": results_dir,
        "clf": clf,
        "train_embeddings": train_embeddings,
        "train_label_ids": train_label_ids,
        "train_df": train_df,
        "test_embeddings": test_embeddings,
        "test_df": test_df,
        "labels": labels,
    }


def project_to_nd(
    train_embeddings: np.ndarray, test_embeddings: np.ndarray, n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP on train embeddings and project test embeddings into the same space."""
    reducer = UMAP(n_components=n_components, metric="cosine", random_state=SEED, n_neighbors=15, min_dist=0.1)
    train_nd = reducer.fit_transform(train_embeddings)
    test_nd = reducer.transform(test_embeddings)
    return train_nd, test_nd


def plot_embedding_space(artifacts: dict, train_2d: np.ndarray, test_2d: np.ndarray, save_path, title: str) -> None:
    clf = artifacts["clf"]
    labels = artifacts["labels"]
    train_label_ids = artifacts["train_label_ids"]
    test_true_ids = artifacts["test_df"]["label_id"].to_numpy()
    test_pred_ids = clf.predict(artifacts["test_embeddings"])
    correct = test_true_ids == test_pred_ids

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_id in range(len(labels)):
        mask = train_label_ids == cls_id
        ax.scatter(
            train_2d[mask, 0], train_2d[mask, 1],
            s=10, color=_TAB10_HEX[cls_id], alpha=0.30, linewidths=0, zorder=1,
        )
    for cls_id in range(len(labels)):
        mask_c = (test_pred_ids == cls_id) & correct
        mask_w = (test_pred_ids == cls_id) & ~correct
        ax.scatter(
            test_2d[mask_c, 0], test_2d[mask_c, 1],
            s=32, color=_TAB10_HEX[cls_id], marker="o", edgecolors="black", linewidths=0.5, zorder=2,
        )
        ax.scatter(
            test_2d[mask_w, 0], test_2d[mask_w, 1],
            s=55, color=_TAB10_HEX[cls_id], marker="X", edgecolors="black", linewidths=0.6, zorder=3,
        )

    class_handles = [Patch(facecolor=_TAB10_HEX[i], label=label) for i, label in enumerate(labels)]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=6, label="train"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=7, label="test (correct)"),
        Line2D([0], [0], marker="X", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=8, label="test (misclassified)"),
    ]
    legend1 = ax.legend(handles=class_handles, title="Decisao class", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=9)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=marker_handles, title="point type", loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8, title_fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(legend1, legend2))
    plt.close(fig)


def plot_and_report_neighbors(
    artifacts: dict, query_idx: int, train_2d: np.ndarray, test_2d: np.ndarray, save_path, title_prefix: str,
) -> None:
    clf = artifacts["clf"]
    labels = artifacts["labels"]
    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]

    query_embedding = artifacts["test_embeddings"][query_idx : query_idx + 1]
    distances, neighbor_idx = clf.kneighbors(query_embedding, n_neighbors=clf.n_neighbors)
    distances, neighbor_idx = distances[0], neighbor_idx[0]

    inv = 1.0 / np.clip(distances, 1e-12, None)
    vote_share = inv / inv.sum()

    true_id = int(test_df.loc[query_idx, "label_id"])
    pred_id = int(clf.predict(query_embedding)[0])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(train_2d[:, 0], train_2d[:, 1], s=8, color="#cccccc", alpha=0.35, linewidths=0, zorder=1)
    for ti, dist in zip(neighbor_idx, distances):
        cls_id = int(artifacts["train_label_ids"][ti])
        ax.plot(
            [test_2d[query_idx, 0], train_2d[ti, 0]], [test_2d[query_idx, 1], train_2d[ti, 1]],
            color="black", linewidth=0.8, alpha=0.6, zorder=2,
        )
        ax.scatter(*train_2d[ti], s=100, color=_TAB10_HEX[cls_id], edgecolors="black", linewidths=1.0, zorder=3)
    ax.scatter(
        *test_2d[query_idx], s=260, marker="*", color=_TAB10_HEX[pred_id], edgecolors="black", linewidths=1.2, zorder=4,
    )

    class_handles = [Patch(facecolor=_TAB10_HEX[i], label=label) for i, label in enumerate(labels)]
    marker_handles = [
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=12, label="query (test)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=8, label=f"k={clf.n_neighbors} nearest train neighbors"),
    ]
    legend1 = ax.legend(handles=class_handles, title="Decisao class", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=9)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=marker_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8)

    correctness = "correct" if pred_id == true_id else "MISCLASSIFIED"
    ax.set_title(
        f"{title_prefix}\ntest #{query_idx} -- true: {labels[true_id]} | predicted: {labels[pred_id]} ({correctness})",
        fontsize=10,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(legend1, legend2))
    plt.close(fig)

    import_id = test_df.loc[query_idx, "ImportJurisprudenciaId"]
    print(f"\nquery: test #{query_idx} (ImportJurisprudenciaId={import_id})")
    print(f"true label:      {labels[true_id]}")
    print(f"predicted label: {labels[pred_id]}  ({correctness})")
    print(f"Sumario: {str(test_df.loc[query_idx, 'Sumario'])[:200]}...\n")
    print(f"{'#':<3}{'train_idx':<10}{'cos_dist':<10}{'vote_share':<12}{'label':<28}{'text'}")
    for rank, (ti, dist, share) in enumerate(zip(neighbor_idx, distances, vote_share), start=1):
        lbl = labels[int(artifacts["train_label_ids"][ti])]
        snippet = str(train_df.loc[ti, "Sumario"])[:100].replace("\n", " ")
        print(f"{rank:<3}{ti:<10}{dist:<10.4f}{share:<12.3f}{lbl:<28}{snippet}")


def build_interactive_embedding_space(artifacts: dict, coords_train: np.ndarray, coords_test: np.ndarray, dims: int, title: str) -> go.Figure:
    """Interactive Plotly version of plot_embedding_space."""
    labels = artifacts["labels"]
    train_label_ids = artifacts["train_label_ids"]
    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]
    test_pred_ids = artifacts["clf"].predict(artifacts["test_embeddings"])
    correct = test_df["label_id"].to_numpy() == test_pred_ids
    scatter = go.Scatter3d if dims == 3 else go.Scatter

    fig = go.Figure()
    for cls_id, label in enumerate(labels):
        mask = train_label_ids == cls_id
        if not mask.any():
            continue
        coords = coords_train[mask]
        hover = [f"[train] {label}<br>{_snippet(t)}" for t in train_df.loc[mask, "Sumario"]]
        kwargs = dict(
            x=coords[:, 0], y=coords[:, 1], mode="markers", name=f"{label} (train)", legendgroup=label,
            marker=dict(size=3 if dims == 3 else 5, color=_TAB10_HEX[cls_id], opacity=0.35),
            text=hover, hoverinfo="text",
        )
        if dims == 3:
            kwargs["z"] = coords[:, 2]
        fig.add_trace(scatter(**kwargs))

    for cls_id, label in enumerate(labels):
        for is_correct, symbol, tag in ((True, "circle", "correct"), (False, "x", "MISCLASSIFIED")):
            mask = (test_pred_ids == cls_id) & (correct == is_correct)
            if not mask.any():
                continue
            coords = coords_test[mask]
            rows = test_df.loc[mask]
            hover = [
                f"[test] true: {labels[int(row.label_id)]} | pred: {label} ({tag})<br>{_snippet(row.Sumario)}"
                for row in rows.itertuples()
            ]
            kwargs = dict(
                x=coords[:, 0], y=coords[:, 1], mode="markers", name=f"{label} (test, {tag})", legendgroup=label,
                marker=dict(size=6 if dims == 3 else 7, symbol=symbol, color=_TAB10_HEX[cls_id], line=dict(width=1, color="black")),
                text=hover, hoverinfo="text",
            )
            if dims == 3:
                kwargs["z"] = coords[:, 2]
            fig.add_trace(scatter(**kwargs))

    if dims == 3:
        fig.update_layout(title=title, scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"))
    else:
        fig.update_layout(title=title, xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return fig


def build_interactive_neighbor_figure(
    artifacts: dict, query_idx: int, coords_train: np.ndarray, coords_test: np.ndarray, dims: int, title_prefix: str,
) -> go.Figure:
    """Interactive Plotly version of plot_and_report_neighbors."""
    clf = artifacts["clf"]
    labels = artifacts["labels"]
    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]
    scatter = go.Scatter3d if dims == 3 else go.Scatter

    query_embedding = artifacts["test_embeddings"][query_idx : query_idx + 1]
    distances, neighbor_idx = clf.kneighbors(query_embedding, n_neighbors=clf.n_neighbors)
    distances, neighbor_idx = distances[0], neighbor_idx[0]
    inv = 1.0 / np.clip(distances, 1e-12, None)
    vote_share = inv / inv.sum()

    true_id = int(test_df.loc[query_idx, "label_id"])
    pred_id = int(clf.predict(query_embedding)[0])
    correctness = "correct" if pred_id == true_id else "MISCLASSIFIED"
    qpoint = coords_test[query_idx]

    fig = go.Figure()
    bg_kwargs = dict(
        x=coords_train[:, 0], y=coords_train[:, 1], mode="markers", name="train (background)",
        marker=dict(size=2 if dims == 3 else 3, color="#cccccc", opacity=0.35),
        hoverinfo="skip", showlegend=False,
    )
    if dims == 3:
        bg_kwargs["z"] = coords_train[:, 2]
    fig.add_trace(scatter(**bg_kwargs))

    for rank, (ti, dist, share) in enumerate(zip(neighbor_idx, distances, vote_share), start=1):
        npoint = coords_train[ti]
        line_kwargs = dict(
            x=[qpoint[0], npoint[0]], y=[qpoint[1], npoint[1]], mode="lines",
            line=dict(color="black", width=1), opacity=0.5, hoverinfo="skip", showlegend=False,
        )
        if dims == 3:
            line_kwargs["z"] = [qpoint[2], npoint[2]]
        fig.add_trace(scatter(**line_kwargs))

        cls_id = int(artifacts["train_label_ids"][ti])
        hover = (
            f"neighbor #{rank} -- {labels[cls_id]}<br>cos dist: {dist:.4f} | vote share: {share:.3f}"
            f"<br>{_snippet(train_df.loc[ti, 'Sumario'])}"
        )
        pt_kwargs = dict(
            x=[npoint[0]], y=[npoint[1]], mode="markers", name=f"#{rank} {labels[cls_id]}",
            marker=dict(size=9 if dims == 3 else 11, color=_TAB10_HEX[cls_id], line=dict(width=1.5, color="black")),
            text=[hover], hoverinfo="text",
        )
        if dims == 3:
            pt_kwargs["z"] = [npoint[2]]
        fig.add_trace(scatter(**pt_kwargs))

    query_hover = (
        f"QUERY test #{query_idx}<br>true: {labels[true_id]} | pred: {labels[pred_id]} ({correctness})"
        f"<br>{_snippet(test_df.loc[query_idx, 'Sumario'])}"
    )
    q_kwargs = dict(
        x=[qpoint[0]], y=[qpoint[1]], mode="markers", name="query (test)",
        marker=dict(size=15 if dims == 3 else 18, symbol="diamond", color=_TAB10_HEX[pred_id], line=dict(width=2, color="black")),
        text=[query_hover], hoverinfo="text",
    )
    if dims == 3:
        q_kwargs["z"] = [qpoint[2]]
    fig.add_trace(scatter(**q_kwargs))

    title = f"{title_prefix} -- test #{query_idx}: true {labels[true_id]} | pred {labels[pred_id]} ({correctness})"
    if dims == 3:
        fig.update_layout(title=title, scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"))
    else:
        fig.update_layout(title=title, xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="serafim_900m", help="key into src.config.MODEL_REGISTRY")
    parser.add_argument("--task", default="knn_probe", help="results_dir task subfolder, e.g. knn_probe or finetune_lora_knn")
    parser.add_argument("--dataset-variant", default="sumario")
    parser.add_argument(
        "--query-idx", type=int, default=None,
        help="test-split row index (0-based) to highlight in the neighbor plot; "
        "defaults to the first misclassified test example, or test #0 if none are misclassified",
    )
    parser.add_argument("--dims", type=int, choices=(2, 3), default=2, help="dimensionality of the interactive Plotly plot (the matplotlib PNGs stay 2D regardless)")
    parser.add_argument("--no-show", action="store_true", help="save the interactive HTML files without trying to open them in a browser")
    args = parser.parse_args()

    model_cfg = get_model_config(args.model)
    artifacts = load_knn_artifacts(model_cfg, args.task, args.dataset_variant)
    results_dir = artifacts["results_dir"]
    title_prefix = f"{model_cfg.display_name} -- {args.task}"

    print(f"projecting {len(artifacts['train_embeddings'])} train + {len(artifacts['test_embeddings'])} test "
          f"embeddings to 2D with UMAP (metric=cosine)...")
    train_2d, test_2d = project_to_nd(artifacts["train_embeddings"], artifacts["test_embeddings"], n_components=2)

    space_path = results_dir / "embedding_space.png"
    plot_embedding_space(artifacts, train_2d, test_2d, space_path, f"{title_prefix}: UMAP (cosine) of mean-pooled embeddings")
    print(f"saved {space_path}")

    query_idx = args.query_idx
    if query_idx is None:
        test_true_ids = artifacts["test_df"]["label_id"].to_numpy()
        test_pred_ids = artifacts["clf"].predict(artifacts["test_embeddings"])
        misclassified = np.flatnonzero(test_true_ids != test_pred_ids)
        query_idx = int(misclassified[0]) if len(misclassified) else 0

    neighbor_path = results_dir / f"neighbor_highlight_test{query_idx}.png"
    plot_and_report_neighbors(artifacts, query_idx, train_2d, test_2d, neighbor_path, title_prefix)
    print(f"saved {neighbor_path}")

    if args.dims == 2:
        coords_train, coords_test = train_2d, test_2d
    else:
        print("projecting to 3D for the interactive plot...")
        coords_train, coords_test = project_to_nd(artifacts["train_embeddings"], artifacts["test_embeddings"], n_components=3)

    space_fig = build_interactive_embedding_space(
        artifacts, coords_train, coords_test, args.dims, f"{title_prefix}: UMAP (cosine) of mean-pooled embeddings",
    )
    space_html = results_dir / f"embedding_space_{args.dims}d.html"
    space_fig.write_html(space_html)
    print(f"saved {space_html}")

    neighbor_fig = build_interactive_neighbor_figure(artifacts, query_idx, coords_train, coords_test, args.dims, title_prefix)
    neighbor_html = results_dir / f"neighbor_highlight_test{query_idx}_{args.dims}d.html"
    neighbor_fig.write_html(neighbor_html)
    print(f"saved {neighbor_html}")

    if not args.no_show:
        try:
            space_fig.show()
            neighbor_fig.show()
        except Exception as exc:
            print(f"couldn't auto-open a browser tab ({exc}); open {space_html} / {neighbor_html} manually")


if __name__ == "__main__":
    main()
