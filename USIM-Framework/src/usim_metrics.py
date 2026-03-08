"""
USIM Metric Calculation Module
"""
import numpy as np

def usim_formula(Kd, Sc, Ee, Ef, C, N):
    """USIM = (Kd × Sc × Ee × Ef) / (C × √N)"""
    return (Kd * Sc * Ee * Ef) / (C * np.sqrt(N))
