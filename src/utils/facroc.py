import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trapz_rule(y, x):
    """
    Manual implementation of the trapezoidal rule for integration.
    Avoids deprecation issues with np.trapz in newer NumPy versions.
    """
    dx = x[1:] - x[:-1]
    y_mid = (y[:-1] + y[1:]) / 2.0
    return np.sum(y_mid * dx)

def aucc(partition, dataset, return_rates=False):
    """
    Computes the Area Under the (ROC) Curve for Clustering (AUCC) from scratch.
    
    Parameters:
    - partition: array-like of shape (N,) indicating cluster assignments.
    - dataset: array-like of shape (N, D) containing the sample coordinates.
    - return_rates: boolean, whether to return FPR and TPR for plotting.
    
    Returns:
    - AUCC value (float) or a dictionary containing 'aucc', 'fpr', and 'tpr'.
    """
    partition = np.asarray(partition)
    dataset = np.asarray(dataset)
    n = len(dataset)
    if n <= 1:
        raise ValueError("Dataset must have at least 2 samples.")
    
    # Compute pairwise Euclidean distance matrix
    sq_sum = np.sum(dataset ** 2, axis=1)
    sq_dists = sq_sum[:, np.newaxis] + sq_sum[np.newaxis, :] - 2 * np.dot(dataset, dataset.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    dists = np.sqrt(sq_dists)
    
    # Extract upper triangle (excluding diagonal)
    tri_u = np.triu_indices(n, k=1)
    d_vals = dists[tri_u]
    labels = (partition[tri_u[0]] == partition[tri_u[1]]).astype(int)
    
    # Handle edge case where all pairs are in same cluster or all in different clusters
    if np.sum(labels) == 0 or np.sum(labels) == len(labels):
        if return_rates:
            return {"aucc": 0.5, "fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0])}
        return 0.5
    
    # In AUCC, smaller distance indicates same-cluster (positive class)
    # Thus, scores are negative distances
    scores = -d_vals
    
    # Sort in descending order
    desc_indices = np.argsort(scores)[::-1]
    scores = scores[desc_indices]
    labels = labels[desc_indices]
    
    # Find unique score thresholds to handle ties
    distinct_value_indices = np.where(scores[:-1] - scores[1:] != 0.0)[0]
    threshold_indices = np.r_[distinct_value_indices, labels.size - 1]
    
    # Cumulative sum of TPs and FPs
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    
    tps = tps[threshold_indices]
    fps = fps[threshold_indices]
    
    # Normalized rates
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    
    # Area under curve
    area = trapz_rule(tpr, fpr)
    
    if return_rates:
        return {"aucc": area, "fpr": fpr, "tpr": tpr}
    return area

def interpolate_roc_curve(fpr, tpr, n_grid=40000):
    """
    Interpolates ROC coordinates to a regular grid of size n_grid (default 40,000).
    Replicates stats::approx in R.
    """
    grid_x = np.linspace(0.0, 1.0, n_grid)
    grid_y = np.interp(grid_x, fpr, tpr)
    return {"x": grid_x, "y": grid_y}

def facroc_plot(non_protected_roc, protected_roc, non_protected_label=None,
                protected_label=None, fout=None, facroc_val=None):
    """
    Plots the two ROC curves and colors the area between them grey.
    Replicates facroc_plot and ggplot versions in R.
    """
    plt.figure(figsize=(5, 5))
    
    # Shaded region between the two curves
    plt.fill_between(non_protected_roc["x"], non_protected_roc["y"], protected_roc["y"],
                     color="lightgrey", label="FACROC Area")
    
    # Plot lines
    plt.plot(non_protected_roc["x"], non_protected_roc["y"], color="red", linewidth=1.5,
             label=non_protected_label if non_protected_label else "Non-Protected")
    plt.plot(protected_roc["x"], protected_roc["y"], color="blue", linewidth=1.5,
             label=protected_label if protected_label else "Protected")
    
    # Reference random guess line
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    
    title = f"FACROC = {facroc_val:.4f}" if facroc_val is not None else "FACROC"
    plt.title(title, fontweight="bold")
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if fout:
        os.makedirs(os.path.dirname(os.path.abspath(fout)), exist_ok=True)
        plt.savefig(fout, dpi=300)
    
    # Check if running inside a Jupyter Notebook environment
    is_notebook = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            is_notebook = True
    except ImportError:
        pass

    if not fout or is_notebook:
        plt.show()
    else:
        plt.close()


def compute_facroc(roc_protected, roc_non_protected, protected_label="Female",
                  non_protected_label="Male", show_plot=True, filename=None):
    """
    Computes the FACROC value (Area Between ROC Curves) by taking the absolute 
    difference of the interpolated TPR values and integrating it from 0 to 1.
    
    Parameters:
    - roc_protected: dict, ROC data (fpr, tpr) for protected group.
    - roc_non_protected: dict, ROC data (fpr, tpr) for non-protected group.
    - protected_label: string, label for protected group.
    - non_protected_label: string, label for non-protected group.
    - show_plot: boolean, whether to generate and save/show the plot.
    - filename: string, path to save the generated plot.
    
    Returns:
    - FACROC score (float).
    """
    p_roc = interpolate_roc_curve(roc_protected["fpr"], roc_protected["tpr"])
    np_roc = interpolate_roc_curve(roc_non_protected["fpr"], roc_non_protected["tpr"])
    
    # Area between curves is integral of absolute difference
    diff = np.abs(np_roc["y"] - p_roc["y"])
    facroc_score = trapz_rule(diff, p_roc["x"])
    
    if show_plot:
        facroc_plot(np_roc, p_roc, non_protected_label, protected_label, fout=filename, facroc_val=facroc_score)
        
    return facroc_score
