import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import datashader as ds
import datashader.transfer_functions as tf

from datashader import reductions
from data_util.calculate_stats import compute_r2, compute_mae, compute_rmse
from scipy.stats import spearmanr
from matplotlib.cm import viridis

def plot_learning_curve(save_path, training_loss, validation_loss):

    epochs = np.arange(1, len(training_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_loss, label='Training Loss', linewidth=2)
    plt.plot(epochs, validation_loss, label='Validation Loss', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')

    # Set x-axis ticks every 10 epochs, including the first and last
    step = 10
    xticks = np.arange(0, len(epochs) + step, step)
    if xticks[-1] != len(epochs):  # Ensure last epoch is included
        xticks = np.append(xticks, len(epochs))
    plt.xticks(xticks)

    plt.grid(False)  # Remove grid
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def parity_plot(
    df,
    save_path,
    plot_title,
    true_column,
    pred_column,
    quantity_label="Target",
    percentile_clip=99.5,
    plot_width=800,
    plot_height=800,
    cmap=viridis,
    font_size=14
):

    # Load data
    y_true = df[true_column].values
    y_pred = df[pred_column].values

    # Metrics computed
    r2 = compute_r2(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)

    print(f"R²         = {r2:.5f}")
    print(f"RMSE       = {rmse:.5f}")
    print(f"MAE        = {mae:.5f}")
    print(f"Spearman R = {spearman_rho:.5f}")

    # Clip range at both ends to ignore outliers (targets may be negative)
    combined = np.concatenate([y_true, y_pred])
    min_val = np.percentile(combined, 100 - percentile_clip)
    max_val = np.percentile(combined, percentile_clip)
    if max_val <= min_val:  # degenerate (constant) data
        min_val, max_val = min_val - 1.0, max_val + 1.0

    x_range = (min_val, max_val)
    y_range = (min_val, max_val)

    # Datashader canvas
    cvs = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=x_range,
        y_range=y_range
    )

    agg = cvs.points(df, true_column, pred_column, agg=reductions.count())
    img = tf.shade(agg, cmap=cmap, how="eq_hist")

    # Convert to NumPy and flip vertically for matplotlib
    img_np = np.array(img.to_pil())[::-1]  # Flip top-down

    # Compute max density value for colorbar labeling
    density_max = agg.data.max()

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    ax.imshow(img_np, extent=extent, origin="lower", aspect="equal")

    # Diagonal parity line y = x
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        linewidth=2,
        linestyle="--",
        label="y = x"
    )

    # Write stats onto plot
    stats_text = (
        f"R² = {r2:.5f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"MAE = {mae:.2f}\n"
        f"ρ = {spearman_rho:.5f}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=font_size,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    ax.set_xlabel(f"True {quantity_label}", fontsize=font_size)
    ax.set_ylabel(f"Predicted {quantity_label}", fontsize=font_size)
    ax.set_title(plot_title, fontsize=font_size + 2)
    ax.legend()

    # Colorbar via a ScalarMappable (the image itself is a pre-shaded RGB array)
    norm = mpl.colors.Normalize(vmin=0, vmax=density_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size / 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
