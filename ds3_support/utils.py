from matplotlib import cm
import numpy as np
import torch
import torchvision


def as_tensors(X, *rest):
    "Calls as_tensor on a bunch of args, all of the first's device and dtype."
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title=None,
    cmap=cm.Blues,
    rotation=45,
    **fig_kwargs
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(**fig_kwargs)
    ax.grid(False)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    if rotation:
        plt.setp(
            ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor"
        )

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def pil_grid(X, **kwargs):
    return torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(X, **kwargs))
