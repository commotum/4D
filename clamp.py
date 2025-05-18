def generate_even_intervals(n):
    """
    Generate n even intervals between -1 and 1, returning n+1 values.
    """
    import numpy as np
    # np.linspace(start, stop, num) returns num evenly spaced samples, inclusive.
    return np.linspace(-1.0, 1.0, n + 1)

n = 9
intervals = generate_even_intervals(n)
print(intervals)

