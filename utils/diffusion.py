import numpy as np
import matplotlib.pyplot as plt

# 定义模块中的函数
def generate_lts_centers(samples, center_num, method='average'):
    """Generate LTS centers based on the provided method."""
    sample_min = np.min(samples)
    sample_max = np.max(samples)
    if method == 'average':
        value_range = np.linspace(sample_min, sample_max, center_num + 1)
        return (value_range[:-1] + value_range[1:]) / 2
    elif method == 'direct':
        return np.linspace(sample_min, sample_max, center_num)

def gaussian_diffusion(x, mean, std_dev):
    """Calculate Gaussian diffusion for membership computation."""
    return np.exp(-0.5 * ((x - mean) / std_dev)**2) / (std_dev * np.sqrt(2 * np.pi))

def calculate_membership(samples, lts_centers, std_dev):
    """Calculate membership values for each sample across LTS centers."""
    memberships = np.zeros((len(samples), len(lts_centers)))
    for i, sample in enumerate(samples):
        for j, center in enumerate(lts_centers):
            memberships[i, j] = gaussian_diffusion(sample, center, std_dev)
        memberships[i, :] /= memberships[i, :].sum()
    return memberships

def prune_and_normalize_memberships(memberships, lts_labels, threshold):
    """Prune and normalize membership values based on the threshold."""
    pruned_results = []
    for membership in memberships:
        pruned = {label: score for label, score in zip(lts_labels, membership) if score > threshold}
        total = sum(pruned.values())
        normalized = {label: np.round(score / total, 2) for label, score in pruned.items()}
        pruned_results.append(normalized)
    return pruned_results

def visualize_memberships(samples, memberships, lts_labels, method):
    """Visualize the membership distribution."""
    plt.figure(figsize=(10, 5))
    for i, sample in enumerate(samples):
        plt.plot(lts_labels, memberships[i], marker='o', label=f'Sample {sample}')
    plt.title(f'Membership Distribution using {method} method')
    plt.xlabel('LTS Labels')
    plt.ylabel('Membership Value')
    plt.legend()
    plt.grid(True)
    plt.show()
