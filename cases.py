from utils.PLTS import *
from utils.PT import *
from utils.diffusion import *


## diffusion
samples = np.array([900, 600, 800, 720]) # COST attribute sample
lts_labels = ['s_-3', 's_-2', 's_-1', 's0', 's1', 's2', 's_3'][::-1] # for cost only(Reverse order)
std_dev_ratio = 0.3
sample_std = np.std(samples)
std_dev = sample_std * std_dev_ratio
prune_threshold = 0.09
method_choice = 'direct'
center_num = len(lts_labels)
lts_centers = generate_lts_centers(samples, center_num, method=method_choice)
memberships = calculate_membership(samples, lts_centers, std_dev)
pruned_plts = prune_and_normalize_memberships(memberships, lts_labels, prune_threshold)
visualize_memberships(samples, memberships, lts_labels, method_choice)
for i, result in enumerate(pruned_plts):
    print(f"Sample {samples[i]}: {result}")



## data
plts_matrix = [
    # A 高强度安保部署
    [
        PLTS({"s3": 0.5, "s2": 0.3, "s1": 0.2}),
        PLTS({'s_-2': 0.24, 's_-3': 0.76}),
	    PLTS({"s-2": 0.6, "s-1": 0.2, "s0": 0.2}),
	    PLTS({"s-1": 0.5, "s-2": 0.4, "s0": 0.1}),
	    PLTS({"s1": 0.8, "s0": 0.2})
    ],
    #B 中度安保部署
    [
        PLTS({"s2": 0.4, "s1": 0.4, "s0": 0.2}),
        PLTS({'s_3': 0.76, 's2': 0.24}),
        PLTS({"s0": 0.6, "s1": 0.2, "s-1": 0.2}),
        PLTS({"s1": 0.5, "s0": 0.3, "s2": 0.2}),
        PLTS({"s2": 0.4, "s1": 0.4, "s0": 0.2})
    ],
    #C 技术驱动方案
    [
        PLTS({"s3": 0.7, "s2": 0.2, "s1": 0.1}),
        PLTS({'s0': 0.19, 's_-1': 0.61, 's_-2': 0.19}),
        PLTS({"s-1": 0.8, "s-2": 0.2}),
        PLTS({"s1": 0.5, "s0": 0.3, "s-1": 0.2}),
        PLTS({"s3": 0.8, "s2": 0.1, "s1": 0.1})
    ],
    # D：环保技术方案
    [
        PLTS({"s2": 0.4, "s1": 0.3, "s0": 0.3}),
        PLTS({'s1': 0.56, 's0': 0.44}),
        PLTS({"s1": 0.6, "s0": 0.3, "s-1": 0.1}),
        PLTS({"s0": 0.4, "s1": 0.6}),
        PLTS({"s-2": 0.5, "s-1": 0.4, "s1": 0.1})
    ]
]

# laguage scale =7
lts_n = 7

# 权重
weights = [0.3, 0.2, 0.25, 0.10, 0.15]  # Weights for each criterion