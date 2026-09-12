from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import MDS


CLASS_COLORS = ["#2A6FDB", "#E4572E"]


def _subplot_grid(n_panels, n_columns=3, panel_size=(5.0, 4.2)):
    n_rows = ceil(n_panels / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows),
        constrained_layout=True,
    )
    return fig, np.atleast_1d(axes).ravel()


def plot_classifier_grid(
    models,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    grid_resolution=80,
):
    """Plot continuous decision functions for scikit-learn classifiers."""
    fig, axes = _subplot_grid(len(models))

    for ax, (name, model) in zip(axes, models.items()):
        DecisionBoundaryDisplay.from_estimator(
            model,
            X_train,
            response_method="decision_function",
            grid_resolution=grid_resolution,
            cmap="RdBu_r",
            alpha=0.35,
            ax=ax,
        )

        sns.scatterplot(
            x=X_train[:, 0],
            y=X_train[:, 1],
            hue=y_train,
            palette=CLASS_COLORS,
            edgecolor="black",
            linewidth=0.45,
            s=48,
            legend=False,
            ax=ax,
        )

        title = name
        if X_test is not None and y_test is not None:
            title += f"\ntest accuracy = {model.score(X_test, y_test):.3f}"

        ax.set_title(title)
        ax.set_xlabel("scaled petal length")
        ax.set_ylabel("scaled petal width")

    for ax in axes[len(models):]:
        ax.set_visible(False)

    return fig, axes


def kernel_diagnostics(kernel_matrix, labels, tolerance=1e-10):
    """Return empirical geometry diagnostics for a fidelity kernel."""
    K = np.asarray(kernel_matrix, dtype=float)
    y = 2 * np.asarray(labels, dtype=float) - 1

    K = 0.5 * (K + K.T)
    eigenvalues = np.clip(np.linalg.eigvalsh(K), 0.0, None)

    numerical_rank = int(np.sum(eigenvalues > tolerance))
    effective_rank = eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()

    target_kernel = np.outer(y, y)
    alignment = np.sum(K * target_kernel) / (
        np.linalg.norm(K, "fro") * np.linalg.norm(target_kernel, "fro")
    )

    identity_mask = np.eye(len(y), dtype=bool)
    same_class = y[:, None] == y[None, :]
    different_class = ~same_class

    mean_off_diagonal = K[~identity_mask].mean()
    mean_within_class = K[same_class & ~identity_mask].mean()
    mean_between_class = K[different_class].mean()

    positive = eigenvalues[eigenvalues > tolerance]
    condition_number = (
        np.inf
        if numerical_rank < len(eigenvalues)
        else positive.max() / positive.min()
    )

    return {
        "numerical rank": numerical_rank,
        "effective rank": effective_rank,
        "target alignment": alignment,
        "mean off-diagonal fidelity": mean_off_diagonal,
        "within-class fidelity": mean_within_class,
        "between-class fidelity": mean_between_class,
        "within-minus-between": mean_within_class - mean_between_class,
        "condition number": condition_number,
    }


def plot_kernel_matrices(kernel_matrices, labels):
    """Plot Gram matrices after sorting samples by class label."""
    order = np.argsort(labels)
    fig, axes = _subplot_grid(len(kernel_matrices))

    for ax, (name, K) in zip(axes, kernel_matrices.items()):
        sns.heatmap(
            np.asarray(K)[np.ix_(order, order)],
            vmin=0.0,
            vmax=1.0,
            cmap="mako",
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("samples sorted by class")
        ax.set_ylabel("samples sorted by class")

    for ax in axes[len(kernel_matrices):]:
        ax.set_visible(False)

    return fig, axes


def plot_kernel_geometry(kernel_matrices, labels, random_state=7):
    """Embed Fubini-Study distances into two dimensions with metric MDS."""
    fig, axes = _subplot_grid(len(kernel_matrices))

    for ax, (name, K) in zip(axes, kernel_matrices.items()):
        fidelity = np.clip(np.asarray(K), 0.0, 1.0)
        distances = np.arccos(np.sqrt(fidelity))

        embedding = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=500,
            normalized_stress="auto",
        ).fit_transform(distances)

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels,
            palette=CLASS_COLORS,
            edgecolor="black",
            linewidth=0.45,
            s=55,
            legend=False,
            ax=ax,
        )

        ax.set_title(name)
        ax.set_xlabel("MDS coordinate 1")
        ax.set_ylabel("MDS coordinate 2")

    for ax in axes[len(kernel_matrices):]:
        ax.set_visible(False)

    return fig, axes


def separator_diagnostics(model, X, grid_resolution=80):
    """Characterize one fitted decision function on a common grid."""
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_resolution)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_resolution)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    surface = model.decision_function(grid).reshape(xx.shape)

    grad_x, grad_y = np.gradient(surface)
    total_variation = np.mean(np.sqrt(grad_x**2 + grad_y**2))

    centered_surface = surface - surface.mean()
    power = np.abs(np.fft.fft2(centered_surface)) ** 2
    power[0, 0] = 0.0

    probabilities = power.ravel()
    if probabilities.sum() > 0:
        probabilities = probabilities / probabilities.sum()
        nonzero = probabilities > 0
        spectral_entropy = -np.sum(
            probabilities[nonzero] * np.log(probabilities[nonzero])
        ) / np.log(probabilities.size)
    else:
        spectral_entropy = 0.0

    return {
        "mean spatial variation": total_variation,
        "spectral entropy": spectral_entropy,
    }


def plot_decision_spectra(models, X, grid_resolution=80):
    """Plot the two-dimensional Fourier power of decision functions."""
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_resolution)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_resolution)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = _subplot_grid(len(models))

    for ax, (name, model) in zip(axes, models.items()):
        surface = model.decision_function(grid).reshape(xx.shape)
        centered_surface = surface - surface.mean()

        power = np.fft.fftshift(np.abs(np.fft.fft2(centered_surface)) ** 2)
        log_power = np.log10(power + 1e-12)

        ax.imshow(log_power, origin="lower", cmap="magma", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("frequency along petal length")
        ax.set_ylabel("frequency along petal width")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(models):]:
        ax.set_visible(False)

    return fig, axes


def vqc_margin(model, encoded_data, batch_size=512):
    """Return p(class 1) - p(class 0) from a fitted Qiskit VQC."""
    margins = []

    for start in range(0, len(encoded_data), batch_size):
        batch = encoded_data[start : start + batch_size]
        probabilities = np.asarray(
            model.neural_network.forward(batch, model.weights)
        ).reshape(len(batch), -1)

        if probabilities.shape[1] != 2:
            raise ValueError("This visualization expects a binary VQC.")

        margins.append(probabilities[:, 1] - probabilities[:, 0])

    return np.concatenate(margins)


def plot_vqc_boundaries(
    models,
    encoders,
    X,
    y,
    X_test=None,
    y_test=None,
    grid_resolution=55,
):
    """Plot continuous VQC probability margins in the original 2D input space."""
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_resolution)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_resolution)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = _subplot_grid(len(models))

    for ax, (name, model) in zip(axes, models.items()):
        encoded_grid = encoders[name].transform(grid)
        margin = vqc_margin(model, encoded_grid).reshape(xx.shape)

        ax.contourf(
            xx,
            yy,
            margin,
            levels=np.linspace(-1.0, 1.0, 21),
            cmap="RdBu_r",
            alpha=0.38,
            extend="both",
        )
        ax.contour(xx, yy, margin, levels=[0.0], colors="black", linewidths=1.5)

        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=y,
            palette=CLASS_COLORS,
            edgecolor="black",
            linewidth=0.45,
            s=48,
            legend=False,
            ax=ax,
        )

        title = name
        if X_test is not None and y_test is not None:
            encoded_test = encoders[name].transform(X_test)
            title += f"\ntest accuracy = {model.score(encoded_test, y_test):.3f}"

        ax.set_title(title)
        ax.set_xlabel("scaled age")
        ax.set_ylabel("scaled salary")

    for ax in axes[len(models):]:
        ax.set_visible(False)

    return fig, axes


def plot_loss_histories(histories):
    """Plot every optimizer restart, grouped by encoding."""
    names = list(dict.fromkeys(name for name, _ in histories))
    palette = dict(zip(names, sns.color_palette("tab10", n_colors=len(names))))

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    for (name, restart), values in histories.items():
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            color=palette[name],
            alpha=0.38,
            linewidth=1.2,
        )

    for name in names:
        matching = [
            values
            for (history_name, _), values in histories.items()
            if history_name == name
        ]
        longest = max(len(values) for values in matching)
        padded = np.full((len(matching), longest), np.nan)

        for row, values in enumerate(matching):
            padded[row, : len(values)] = values

        ax.plot(
            np.arange(1, longest + 1),
            np.nanmedian(padded, axis=0),
            color=palette[name],
            linewidth=2.7,
            label=name,
        )

    ax.set_xlabel("objective-function evaluation")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("VQC optimization histories: thin lines are individual restarts")
    ax.legend(ncol=2)

    return fig, ax