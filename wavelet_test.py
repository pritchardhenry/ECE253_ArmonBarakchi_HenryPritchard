import pywt
import matplotlib.pyplot as plt
import numpy as np

# Load the wavelet
wavelet = pywt.Wavelet('db2')

# Get wavelet and scaling functions (discrete approximation)
phi, psi, x = wavelet.wavefun(level=10)  # level controls resolution

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, phi)
plt.title("Scaling function (phi) - db2")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, psi)
plt.title("Wavelet function (psi) - db2")
plt.grid(True)

plt.tight_layout()
plt.show()