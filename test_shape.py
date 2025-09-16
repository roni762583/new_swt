
import torch
from kymatio.torch import Scattering1D

# Parameters
T = 256   # input length
J = 6
Q = 4

scat = Scattering1D(J=J, Q=Q, shape=T)
x = torch.randn(1, T)   # dummy input

Sx = scat(x)
print("Raw scattering output shape:", Sx.shape)

# If time-averaged (common for feature vectors)  
features = Sx.mean(-1)
print("Time-averaged feature shape:", features.shape)
print("Number of WST features:", features.shape[1])

print("
WST Configuration:")
print(f"  Input length (T): {T}")
print(f"  Scales (J): {J}")
print(f"  Wavelets per octave (Q): {Q}")
print(f"  Output features: {features.shape[1]}")
