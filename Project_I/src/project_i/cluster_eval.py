import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class ClusterEvaluator:
    """Standardized evaluation of clustering results.

    Usage:
        evaluator = ClusterEvaluator(X, labels, dates, name="Daily profiles")
        evaluator.print_scores()
        evaluator.plot_all()

        # Or for K selection:
        ClusterEvaluator.plot_k_selection(X, k_range=range(2, 16))
    """

    def __init__(self, X, labels, dates, name="Clustering"):
        """
        Parameters
        ----------
        X : np.ndarray — feature matrix (n_samples, n_features)
        labels : np.ndarray — cluster labels (n_samples,)
        dates : array-like — date/datetime index for each sample
        name : str — experiment name for plot titles
        """
        self.X = np.asarray(X)
        self.labels = np.asarray(labels)
        self.dates = pd.DatetimeIndex(dates)
        self.name = name
        self.n_clusters = len(np.unique(labels))
        self._scores = None

        # Consistent color mapping: each unique label gets a fixed color
        unique_labels = sorted(np.unique(self.labels))
        base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
        self.color_map = {label: base_colors[i] for i, label in enumerate(unique_labels)}

    def scores(self):
        """Compute and cache clustering quality scores."""
        if self._scores is not None:
            return self._scores

        sil = silhouette_score(self.X, self.labels)
        ch = calinski_harabasz_score(self.X, self.labels)
        db = davies_bouldin_score(self.X, self.labels)

        # Per-cluster silhouette
        sil_samples = silhouette_samples(self.X, self.labels)
        per_cluster_sil = {}
        for c in sorted(np.unique(self.labels)):
            per_cluster_sil[c] = sil_samples[self.labels == c].mean()

        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        sizes = dict(zip(unique, counts))
        pcts = {k: v / len(self.labels) * 100 for k, v in sizes.items()}

        self._scores = {
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "per_cluster_silhouette": per_cluster_sil,
            "cluster_sizes": sizes,
            "cluster_pcts": pcts,
        }
        return self._scores

    def print_scores(self):
        """Print a summary table of all scores."""
        s = self.scores()
        print(f"\n{'=' * 60}")
        print(f"  {self.name} — K={self.n_clusters}")
        print(f"{'=' * 60}")
        print(f"  Silhouette score:       {s['silhouette']:.4f}")
        print(f"  Calinski-Harabasz:      {s['calinski_harabasz']:.1f}")
        print(f"  Davies-Bouldin:         {s['davies_bouldin']:.4f}")
        print(f"\n  Per-cluster breakdown:")
        print(f"  {'Cluster':>8} {'Size':>6} {'%':>7} {'Silhouette':>11}")
        print(f"  {'-' * 35}")
        for c in sorted(s["cluster_sizes"].keys()):
            print(f"  {c:>8} {s['cluster_sizes'][c]:>6} {s['cluster_pcts'][c]:>6.1f}% "
                  f"{s['per_cluster_silhouette'][c]:>10.4f}")
        print()

    @staticmethod
    def plot_k_selection(X, k_range=range(2, 16), name=""):
        """Elbow plot (inertia) and silhouette score vs K."""
        inertias = []
        sil_scores = []

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, km.labels_))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=6)
        ax1.set_xlabel("K (number of clusters)")
        ax1.set_ylabel("Inertia")
        ax1.set_title(f"Elbow Plot{' — ' + name if name else ''}")
        ax1.set_xticks(list(k_range))

        ax2.plot(list(k_range), sil_scores, "ro-", linewidth=2, markersize=6)
        ax2.set_xlabel("K (number of clusters)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title(f"Silhouette Score vs K{' — ' + name if name else ''}")
        ax2.set_xticks(list(k_range))

        plt.tight_layout()
        plt.show()

        return dict(zip(k_range, sil_scores))

    def plot_profiles(self, feature_labels=None):
        """Mean profile per cluster with std band."""
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(5 * self.n_clusters, 4),
                                 sharey=True, squeeze=False)
        axes = axes.flatten()

        x_vals = np.arange(self.X.shape[1])
        if feature_labels is not None:
            x_labels = feature_labels
        else:
            x_labels = x_vals

        for i, c in enumerate(sorted(np.unique(self.labels))):
            mask = self.labels == c
            cluster_data = self.X[mask]
            mean = cluster_data.mean(axis=0)
            std = cluster_data.std(axis=0)
            color = self.color_map[c]

            axes[i].plot(x_vals, mean, color=color, linewidth=2)
            axes[i].fill_between(x_vals, mean - std, mean + std, alpha=0.2, color=color)
            axes[i].set_title(f"Cluster {c} (n={mask.sum()})")
            axes[i].set_xlabel("Feature")

            if len(x_vals) == 24:
                axes[i].set_xlabel("Hour")
                axes[i].set_xticks(range(0, 24, 3))
            elif len(x_vals) == 168:
                for d in range(7):
                    axes[i].axvline(d * 24, color="gray", linestyle=":", alpha=0.3)
                axes[i].set_xlabel("Mon → Sun (hourly)")
                axes[i].set_xticks([d * 24 + 12 for d in range(7)])
                axes[i].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                        fontsize=8)

        axes[0].set_ylabel("Power [kW]")
        plt.suptitle(f"Cluster Profiles — {self.name}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_timeline(self):
        """One row per year — solid colored bars for each sample's date range."""
        from matplotlib.patches import Patch

        unique_labels = sorted(np.unique(self.labels))

        years = sorted(set(self.dates.year))
        n_years = len(years)

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

        fig, axes = plt.subplots(n_years, 1, figsize=(20, 0.7 * n_years + 2),
                                 sharex=True, squeeze=False)
        axes = axes.flatten()

        # Determine bar width from data spacing
        if len(self.dates) > 1:
            diffs = np.diff(sorted(self.dates))
            median_gap = np.median([d.days for d in diffs if hasattr(d, 'days')])
            bar_width = max(median_gap, 1)
        else:
            bar_width = 7

        for i, year in enumerate(years):
            ax = axes[i]
            year_mask = self.dates.year == year
            year_dates = self.dates[year_mask]
            year_labels = self.labels[year_mask]

            doy = year_dates.dayofyear

            # Draw filled bars spanning the full row height
            for d, label in zip(doy, year_labels):
                ax.barh(0, bar_width, left=d, height=1, color=self.color_map[label],
                        edgecolor="none", linewidth=0)

            ax.set_yticks([])
            ax.set_ylabel(str(year), fontsize=10, rotation=0, labelpad=30, va="center")
            ax.set_xlim(1, 366)
            ax.set_ylim(-0.5, 0.5)

            # Month grid lines
            for ms in month_starts[1:]:
                ax.axvline(ms, color="white", linewidth=0.8, alpha=0.7)

            ax.set_xticks(month_starts)
            ax.set_xticklabels(month_names if i == n_years - 1 else [], fontsize=9)
            ax.tick_params(axis="x", length=3)
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Legend
        patches = [Patch(facecolor=self.color_map[c], label=f"Cluster {c}")
                   for c in unique_labels]
        fig.legend(handles=patches, loc="upper right", fontsize=9, ncol=self.n_clusters,
                   frameon=True, edgecolor="gray")
        plt.suptitle(f"Cluster Timeline — {self.name}", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.show()

    def plot_distribution(self):
        """Bar chart of cluster sizes."""
        s = self.scores()
        clusters = sorted(s["cluster_sizes"].keys())
        sizes = [s["cluster_sizes"][c] for c in clusters]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar([f"Cluster {c}" for c in clusters], sizes,
                      color=[self.color_map[c] for c in clusters])
        for bar, size, pct in zip(bars, sizes, [s["cluster_pcts"][c] for c in clusters]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{size}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"Cluster Sizes — {self.name}")
        plt.tight_layout()
        plt.show()

    def plot_projection(self, method="pca"):
        """2D scatter via PCA or t-SNE, colored by cluster."""
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(self.X)
            x_label = f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})"
            y_label = f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})"
        else:
            perplexity = min(30, len(self.X) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = reducer.fit_transform(self.X)
            x_label = "t-SNE 1"
            y_label = "t-SNE 2"

        fig, ax = plt.subplots(figsize=(8, 6))

        for c in sorted(np.unique(self.labels)):
            mask = self.labels == c
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[self.color_map[c]], s=15,
                       label=f"Cluster {c}", alpha=0.6, edgecolors="none")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"2D Projection ({method.upper()}) — {self.name}")
        ax.legend(markerscale=2, fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_all(self, feature_labels=None, projection="pca"):
        """Run all plots in standard order."""
        self.print_scores()
        self.plot_profiles(feature_labels=feature_labels)
        self.plot_timeline()
        self.plot_distribution()
        self.plot_projection(method=projection)
