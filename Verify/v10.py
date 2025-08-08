import numpy as np
import random   

# 1. Generate a single pair of random 4 vectors q and p.
# The values are uniformly distributed between -0.6 and 0.6.
q = (0.6 - (-0.6)) * np.random.rand(4) - 0.6
p = (0.6 - (-0.6)) * np.random.rand(4) - 0.6
print(f"Random Query Vector (q): [{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f}]")
print(f"Random Key Vector   (p): [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f} {p[3]:+.3f}]\n")

