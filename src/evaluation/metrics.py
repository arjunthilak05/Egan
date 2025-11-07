"""
Evaluation Metrics for QGAN Performance

Implements:
- Mode coverage (key metric for our contribution)
- Sample diversity
- Inception Score (IS)
- Fréchet Inception Distance (FID)
- Quality metrics
"""

import torch
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Optional
import warnings


def mode_coverage_score(
    generated_samples: np.ndarray,
    true_modes: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, np.ndarray]:
    """
    Calculate mode coverage: percentage of true modes captured by generator.

    THIS IS A KEY METRIC FOR OUR PAPER: We claim that entropy regularization
    improves mode coverage and prevents mode collapse.

    Args:
        generated_samples: Generated samples [n_samples, dim]
        true_modes: True mode centers [n_modes, dim]
        threshold: Distance threshold to consider a mode as "covered"

    Returns:
        (coverage_percentage, covered_modes_mask)
    """
    n_modes = len(true_modes)

    # For each true mode, find the closest generated sample
    distances = cdist(true_modes, generated_samples, metric='euclidean')
    min_distances = distances.min(axis=1)

    # A mode is "covered" if at least one sample is within threshold distance
    covered = min_distances < threshold
    coverage = covered.sum() / n_modes

    return float(coverage), covered


def sample_diversity_score(samples: np.ndarray, k: int = 5) -> float:
    """
    Measure diversity of generated samples using k-nearest neighbor distances.

    Higher diversity = samples are more spread out.

    Args:
        samples: Generated samples [n_samples, dim]
        k: Number of nearest neighbors

    Returns:
        Average distance to k-th nearest neighbor (higher = more diverse)
    """
    if len(samples) < k + 1:
        return 0.0

    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(samples)
    distances, _ = nbrs.kneighbors(samples)

    # Exclude self (distance 0) and compute mean of k-NN distances
    diversity = distances[:, 1:].mean()

    return float(diversity)


def pairwise_distance_diversity(samples: np.ndarray) -> float:
    """
    Compute mean pairwise distance between samples.

    Args:
        samples: Generated samples [n_samples, dim]

    Returns:
        Mean pairwise distance
    """
    # Compute pairwise distances
    distances = cdist(samples, samples, metric='euclidean')

    # Take upper triangle (exclude diagonal and duplicates)
    n = len(samples)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_distance = distances[mask].mean()

    return float(mean_distance)


def mode_collapse_metric(
    generated_samples: np.ndarray,
    n_clusters: int = 10,
    threshold: float = 0.1
) -> float:
    """
    Detect mode collapse by clustering generated samples.

    If samples cluster into fewer modes than expected, mode collapse occurred.

    Args:
        generated_samples: Generated samples [n_samples, dim]
        n_clusters: Expected number of modes
        threshold: Minimum fraction of samples in a cluster to be considered valid

    Returns:
        Collapse score (0 = no collapse, 1 = complete collapse)
    """
    from sklearn.cluster import KMeans

    # Cluster generated samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(generated_samples)

    # Count samples per cluster
    unique, counts = np.unique(labels, return_counts=True)
    fractions = counts / len(generated_samples)

    # Count "valid" clusters (with sufficient samples)
    valid_clusters = (fractions > threshold).sum()

    # Collapse score: 1 - (valid_clusters / expected_clusters)
    collapse = 1.0 - (valid_clusters / n_clusters)

    return float(np.clip(collapse, 0, 1))


def inception_score(
    samples: torch.Tensor,
    classifier: torch.nn.Module,
    n_splits: int = 10,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS) for generated samples.

    IS = exp(E[KL(p(y|x) || p(y))])

    Higher IS = better quality and diversity.

    Args:
        samples: Generated samples [n_samples, ...]
        classifier: Pre-trained classifier (e.g., Inception-v3)
        n_splits: Number of splits for computing mean and std
        batch_size: Batch size for classifier inference

    Returns:
        (mean_IS, std_IS)
    """
    classifier.eval()
    n_samples = len(samples)

    # Get predictions
    all_preds = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = samples[i:i+batch_size]
            preds = torch.softmax(classifier(batch), dim=1)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Split and compute IS for each split
    scores = []
    split_size = n_samples // n_splits

    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i < n_splits - 1 else n_samples

        preds = all_preds[start:end]

        # p(y|x)
        py_given_x = preds

        # p(y) = E_x[p(y|x)]
        py = np.mean(py_given_x, axis=0)

        # KL divergence
        kl = py_given_x * (np.log(py_given_x + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)

        # IS for this split
        is_score = np.exp(np.mean(kl))
        scores.append(is_score)

    return float(np.mean(scores)), float(np.std(scores))


def frechet_inception_distance(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and fake samples.

    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))

    Lower FID = better quality.

    Args:
        real_features: Features from real images [n_real, feature_dim]
        fake_features: Features from generated images [n_fake, feature_dim]
        eps: Epsilon for numerical stability

    Returns:
        FID score
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate squared difference of means
    diff = mu_real - mu_fake
    mean_diff = np.dot(diff, diff)

    # Calculate sqrt of product of covariances
    cov_sqrt, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Check for imaginary numbers (numerical instability)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Calculate trace
    trace = np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)

    fid = mean_diff + trace

    return float(fid)


def quality_diversity_tradeoff(
    generated_samples: np.ndarray,
    real_samples: np.ndarray,
    true_modes: Optional[np.ndarray] = None
) -> dict:
    """
    Comprehensive evaluation combining quality and diversity metrics.

    Args:
        generated_samples: Generated samples
        real_samples: Real samples
        true_modes: True mode centers (if known)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Diversity metrics
    metrics['diversity_knn'] = sample_diversity_score(generated_samples, k=5)
    metrics['diversity_pairwise'] = pairwise_distance_diversity(generated_samples)

    # Mode coverage (if true modes are known)
    if true_modes is not None:
        coverage, _ = mode_coverage_score(generated_samples, true_modes, threshold=0.5)
        metrics['mode_coverage'] = coverage

        # Mode collapse
        n_modes = len(true_modes)
        metrics['mode_collapse'] = mode_collapse_metric(generated_samples, n_clusters=n_modes)

    # Quality: compare distributions
    # Use Wasserstein distance (Earth Mover's Distance)
    from scipy.stats import wasserstein_distance

    # Compare marginal distributions
    for dim in range(min(generated_samples.shape[1], 5)):  # Check first 5 dimensions
        wd = wasserstein_distance(
            real_samples[:, dim],
            generated_samples[:, dim]
        )
        metrics[f'wasserstein_dim_{dim}'] = wd

    return metrics


def entropy_diversity_correlation(
    entropies: np.ndarray,
    samples: np.ndarray
) -> dict:
    """
    Analyze correlation between entanglement entropy and sample diversity.

    THIS IS KEY FOR OUR PAPER: We claim that higher entropy leads to higher diversity.

    Args:
        entropies: Entanglement entropies for each sample [n_samples]
        samples: Generated samples [n_samples, dim]

    Returns:
        Dictionary with correlation metrics
    """
    from scipy.stats import pearsonr, spearmanr

    # Compute diversity for each sample (distance to nearest neighbor)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(samples)
    distances, _ = nbrs.kneighbors(samples)
    diversities = distances[:, 1]  # Distance to nearest neighbor (exclude self)

    # Compute correlations
    pearson_corr, pearson_p = pearsonr(entropies, diversities)
    spearman_corr, spearman_p = spearmanr(entropies, diversities)

    return {
        'pearson_correlation': float(pearson_corr),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_corr),
        'spearman_pvalue': float(spearman_p),
        'mean_entropy': float(np.mean(entropies)),
        'std_entropy': float(np.std(entropies)),
        'mean_diversity': float(np.mean(diversities)),
        'std_diversity': float(np.std(diversities))
    }


class MetricsTracker:
    """
    Track metrics during training for easy plotting and analysis.
    """

    def __init__(self):
        self.metrics = {}

    def add(self, metric_name: str, value: float, step: int):
        """Add a metric value at a specific step."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {'steps': [], 'values': []}

        self.metrics[metric_name]['steps'].append(step)
        self.metrics[metric_name]['values'].append(value)

    def get(self, metric_name: str) -> Tuple[List[int], List[float]]:
        """Get all values for a metric."""
        if metric_name not in self.metrics:
            return [], []

        return self.metrics[metric_name]['steps'], self.metrics[metric_name]['values']

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]['values']:
            return None

        return self.metrics[metric_name]['values'][-1]

    def to_dict(self) -> dict:
        """Export all metrics as a dictionary."""
        return self.metrics
