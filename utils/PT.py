
import numpy as np
def prospect(x, α, β, λ):
    if x >= 0:
        return np.log(1 + α * x)
    else:
        return -λ * np.log(1 - β * x)

