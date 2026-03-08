from usim_metrics import usim_formula
from scaling_utils import scaling_strategy

Kd = 0.18
Sc = 0.71
Ee = 1.0
Ef = 0.0011
C = 0.15
N = 4

usim_score = usim_formula(Kd, Sc, Ee, Ef, C, N)
print(f"USIM Score: {usim_score}")

print(scaling_strategy(N))
