import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import os.path
import pandas as pd


@torch.no_grad()
def levels(level_fn, pnts: np.ndarray, ax: plt.Axes, num_pnts: int = 200,
           num_levels: int = 10, font_size: int = 15):
    """
    Plot boundary of a model, e.g., the confidence of a prediction.

    Args:
        level_fn: classifier layer of the model for making boundaries.
        pnts: Points (embeddings) to draw.
        ax: Current figure axis.
        num_pnts: Number of fake points to generate for creating boundaries.
        num_levels: Number of softmax levels for showing decision boundary.
        font_size: Font size for levels.
    Return:
         None
    """
    # Compute grid params
    _x = np.percentile(pnts[:, 0], q=[0.01, 99.99])
    _y = np.percentile(pnts[:, 1], q=[0.01, 99.99])
    step_x = np.abs(_x[1] - _x[0]) / num_pnts
    step_y = np.abs(_y[1] - _y[0]) / num_pnts

    # Make the gird
    _xx, _yy = np.meshgrid(np.arange(_x[0], _x[1], step_x),
                           np.arange(_y[0], _y[1], step_y))
    pnts_t = torch.from_numpy(np.c_[_xx.ravel(), _yy.ravel()]).to(torch.float32)

    # Compute predictions
    pred = level_fn(pnts_t).softmax(-1).max(-1)[0]
    _zz = pred.numpy().reshape(_xx.shape)

    #
    ax.contourf(_xx, _yy, _zz, levels=num_levels, alpha=0.)
    cs = ax.contour(_xx, _yy, _zz, levels=num_levels, alpha=0.45)
    plt.clabel(cs, inline=1, fontsize=font_size, colors='black')
    return _x, _y


def centers(pnts, y, ax, mapping: dict = None, label_encoder=None,
            font_size: int = 20, delta: float = 0.01):
    """

    Args:
        pnts:
        y:
        ax:
        mapping:
        label_encoder:
    :return:
    """
    #
    for _y in np.unique(y):
        # Compute center and plot it
        center = pnts[_y == y].mean(0)
        sns.scatterplot(*center, c='black', s=100, ax=ax)

        # Add label
        _label = f'Cluster {_y}' if mapping is None \
            else mapping[label_encoder.inverse_transform([_y])[0]]
        ax.text(center[0] - delta, center[1] + delta,
                _label, dict(size=font_size))


def plot_embeddings(embeddings: np.ndarray, y: np.ndarray,
                    clf: torch.nn.Module = None,
                    mapping: dict = None,
                    label_encoder=None,
                    figsize: tuple = (10, 10),
                    title: str = 'Epoch',
                    num: int = 0,
                    out_path: str = None,
                    **kwargs):
    plt.figure(figsize=figsize, clear=True, num=num)
    ax = plt.gca()

    # Make the confidence level is classifier is provided
    _x, _y = None, None
    if clf is not None:
        _x, _y = levels(clf, embeddings, ax)

    #
    # centers(embeddings, y, ax, mapping, label_encoder)

    # Draw points
    vals, cnts = np.unique(y, return_counts=True)
    majority_c = vals[cnts.argmax()]
    for _c in vals:
        _pnts = embeddings[y == _c]
        _label = f'Class {_c}' if mapping is None \
            else mapping[label_encoder.inverse_transform([_c])[0]]

        # Mask for focusing near the decision boundary
        mask = torch.ones(_pnts.shape[0], dtype=bool)
        # if _c == majority_c and _x is not None and _y is not None:
        #     mask[(_pnts[:, 0] < _x[0]) | (_x[1] < _pnts[:, 0])] = False
        #     mask[(_pnts[:, 1] < _y[0]) | (_y[1] < _pnts[:, 1])] = False
        ax.scatter(x=_pnts[mask, 0], y=_pnts[mask, 1],
                   label=_label, color=plt.cm.tab10(_c), s=25, alpha=0.5)

    ax.set_title(title)
    ax.legend()
    # plt.axis('off')
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path + f'.png')
    plt.show()


def plot_feature_distr(dataset_path, features):
    """
    Plot the distribution of all features in a dataset.

    Args:
        dataset_path: path to files to load.
        features: name of the features.
    """
    # Load the chunks
    x, y = [], []
    b_path = os.path.join(dataset_path, 'chunks')
    for f in os.listdir(b_path):
        file_p = os.path.join(b_path, f)
        if os.path.isfile(file_p):
            _x, _y = torch.load(file_p)
            x.append(_x)
            y.append(_y)
    #
    x = torch.cat(x, dim=0)
    print(x.quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0))
    x = x.numpy()
    df = pd.DataFrame(x, columns=features[:-1])
    df.hist(bins=200, figsize=(20, 20))
    plt.tight_layout()
    plt.show()
    #
    eps = x[:, :-2].min(0) + 1
    x[:, :-2] = np.log(x[:, :-2] + eps)
    df = pd.DataFrame(x, columns=features[:-1])
    df.hist(bins=200, figsize=(20, 20))
    plt.tight_layout()
    plt.show()
