"""
Utilities for running a quick demo of the complete inference pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from path import Path

from utils.js_plot_style import BLUE, RED, LIGHT_BLUE, DARK_BLUE, LIGHT_GRAY, apply_style, get_font
from models.ae_ct_multi import AutoencoderCTMulti, DeviceCorrelationInfo
from dataset.adr_preprocessor import PreProcessor, json_load, json_dump

from typing import Optional, Tuple
from einops import rearrange

def plot_ts(
    x: np.ndarray,
    x_rec: np.ndarray,
    y_range: Optional[Tuple[float, float]] = None,
    title: str = "",
    out_path: Optional[str] = None,
) -> None:
    """
    Plot and show the input time series.

    Args:
        x: Time series to plot.
            - shape: (N_samples,)
        x_rec: Reconstruction of the time series `x`.
            - shape: (N_samples,)
        y_range: Y-axis range for the plot.
            - If None, use the default plot range.
        title: Title of the plot.
        out_path: Path to save the plot.
            - If None, the plot will not be saved.
    """
    plt.figure(figsize=(12, 7))

    # plot the original and reconstructed time series
    plt.plot(x[0], color=BLUE, linewidth=6)
    plt.plot(x_rec[0], color=RED, linewidth=2, alpha=0.85)

    if y_range is not None:
        plt.ylim(*y_range)

    # change font properties
    if title != "":
        plt.title(title, fontproperties=get_font(), fontsize=16, color=LIGHT_BLUE)

    apply_style()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.show()
    plt.close()

def plot_self_att_matrix(
    self_att_matrix: np.ndarray,
    labels: list[str],
    colormap: str = "viridis",
    out_path: str | None = None,
    title: str = "Self-Attention Matrix",
    ordered: bool = True
) -> None:
    """
    Plot the self-attention matrix as a heatmap with labels sorted alphabetically.

    Args:
        self_att_matrix: Self-attention matrix (N x N), values in [0, 1].
        labels: Labels for the rows/columns (length N).
        colormap: Colormap for the heatmap.
        out_path: If provided, save the plot to this path.
        title: Title for the heatmap.
    """
    if not ordered:
        # If not ordered, use the original order of labels
        sorted_indices = np.arange(len(labels))
    else:
        # Sort labels and self-attention matrix
        sorted_indices = np.argsort(labels)
    labels_sorted = [labels[i] for i in sorted_indices]
    matrix_sorted = self_att_matrix[np.ix_(sorted_indices, sorted_indices)]

    if ordered:
        title += " (Ordered by Labels)"
        
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    img = ax.imshow(matrix_sorted, cmap=colormap, vmin=0, vmax=1)
    cbar = plt.colorbar(img)

    # Optionally set font properties for colorbar labels
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(get_font())

    # Add title
    ax.set_title(title, fontproperties=get_font(weight="Bold", size=18), pad=15)

    # Set tick labels with rotation for x-axis
    ax.set_xticks(np.arange(len(labels_sorted)))
    ax.set_yticks(np.arange(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, fontproperties=get_font(size=10), rotation=90, ha="right")
    ax.set_yticklabels(labels_sorted, fontproperties=get_font(size=10))

    apply_style()
    plt.tight_layout()
    ax.grid(False)

    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show()
    plt.close()



def inference_demo():
    """
    Quick demo of the complete inference pipeline on a test sample.

    Args:
        test_sample_idx: index of the test sample in the split
        exp_name: name of the experiment you want to test
    """
    
    import yaml
    from argparse import Namespace
    from tqdm import tqdm
    
    cnf = yaml.safe_load(open("configs/adr_ae_ct_multi.yaml", "r"))
    cnf = Namespace(**cnf)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # init model and load weights of the best epoch
    model = torch.load(
            "checkpoints/ae_ct_multi_20_newLabeling_TimeCoherent.pth", map_location=device, weights_only=False
        )

    model.eval()
    model.requires_grad(False)
    
    

    cnf.ds_root = Path(cnf.ds_root)
    # load test split info and select the required sample
    split = json_load(cnf.ds_root / "splits" / cnf.split)["test"][cnf.line]
    
    for idx, group in enumerate(tqdm(split, desc="Processing groups")):

        # build model input
        pp = PreProcessor(
            ds_root=cnf.ds_root,
            split_name=cnf.split,
            patch_size=8,
        )
        x_list = []
        for tag in group:
            x = np.load(cnf.ds_root / "vibrations" / tag)
            x = pp.apply(x)
            x_list.append(x[None, ...])
        x = torch.cat(x_list, dim=0)
        x = x.unsqueeze(0).to(device)
        
        # obtain the reconstruction and self-attention matrix
        x_rec, deviceCorrelationInfo = model.forward(x)
        
        x, x_rec = rearrange(x, 'b d s f -> (b d) s f'), rearrange(x_rec, 'b d s f -> (b d) s f')
        
        err = torch.abs(x - x_rec).view(x.shape[0], -1)  # shape: (B, C*ts_len)
        scores = torch.quantile(err, 0.98, dim=1)
        
        att_matrix = deviceCorrelationInfo.attention_weights.squeeze(0).cpu().numpy()

        out_dir = Path("output") / f"test_sample_{idx}"
        out_dir.makedirs_p()

        # plot the self-attention matrix and reconstructed time series
        plot_self_att_matrix(
            att_matrix, labels=group, colormap="viridis", out_path=out_dir / "att_matrix.png"
        )
        for i in range(x.shape[0]):
            plot_ts(
                x=x.cpu().numpy()[i].reshape((1, -1)),
                x_rec=x_rec.cpu().numpy()[i].reshape((1, -1)),
                y_range=(-0.1, 1),
                title=f"Device {i}: {group[i]} [Score: {scores[i].item():.2f}]",
                out_path=out_dir / f"{i:02d}_{group[i]}.png",
            )


if __name__ == "__main__":
    inference_demo()