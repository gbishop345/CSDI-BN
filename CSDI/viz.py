import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.signal import welch

SEQ_LENGTH = 48
NUM_FEATURES = 35
SAVE_CHOL_PATH = "blue_noise_chol_matrix.pt"

# Load Cholesky decomposition
if not os.path.exists(SAVE_CHOL_PATH):
    raise FileNotFoundError(f"Cholesky factor {SAVE_CHOL_PATH} not found. Run compute_and_save_chol() first.")

L_chol = torch.load(SAVE_CHOL_PATH).numpy()  # Load as NumPy array
flat_dim = SEQ_LENGTH * NUM_FEATURES

# Generate Gaussian noise
z = np.random.randn(flat_dim)  # Uncorrelated noise

# Apply Cholesky transformation
blue_noise = (L_chol @ z).reshape(SEQ_LENGTH, NUM_FEATURES)  # Reshape to (48, 35)

# Generate Gaussian noise for comparison
gaussian_noise = np.random.randn(SEQ_LENGTH, NUM_FEATURES)  # Pure white noise

def compute_fft_spectrum(noise):
    fft_result = fft2(noise)  # Compute 2D FFT
    fft_magnitude = np.abs(fft_result)**2  # Power Spectrum
    return fftshift(fft_magnitude)  # Shift DC component to center

blue_fft = compute_fft_spectrum(blue_noise)
gaussian_fft = compute_fft_spectrum(gaussian_noise)

def radial_spectrum(power_spectrum):
    rows, cols = power_spectrum.shape
    center_x, center_y = rows // 2, cols // 2
    max_radius = int(np.sqrt(center_x**2 + center_y**2))

    radial_bins = np.zeros(max_radius)
    counts = np.zeros(max_radius)

    for x in range(rows):
        for y in range(cols):
            radius = int(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
            if radius < max_radius:
                radial_bins[radius] += power_spectrum[x, y]
                counts[radius] += 1

    return radial_bins / (counts + 1e-6)  # Normalize by count

blue_radial = radial_spectrum(blue_fft)
gaussian_radial = radial_spectrum(gaussian_fft)

def log_spectrum(noise):
    f, Pxx = welch(noise.flatten(), nperseg=len(noise.flatten())//4)
    return f[1:], Pxx[1:]  # Remove DC component (0 Hz)

blue_freqs, blue_power = log_spectrum(blue_noise)
gaussian_freqs, gaussian_power = log_spectrum(gaussian_noise)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 2D FFT Magnitude Spectrum
ax1, ax2 = axes[0]
im1 = ax1.imshow(np.log1p(blue_fft), cmap="inferno", aspect="auto")
ax1.set_title("Blue Noise Frequency Spectrum")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(np.log1p(gaussian_fft), cmap="inferno", aspect="auto")
ax2.set_title("Gaussian Noise Frequency Spectrum")
plt.colorbar(im2, ax=ax2)

# 2. Radial Frequency Distribution
ax3, ax4 = axes[1]
ax3.plot(blue_radial, label="Blue Noise", color="blue")
ax3.plot(gaussian_radial, label="Gaussian Noise", color="orange")
ax3.set_title("Radial Frequency Distribution")
ax3.set_xlabel("Radius")
ax3.set_ylabel("Power")
ax3.legend()

# 3. Log-Log Power Spectrum (Fix: Now correctly inside axes)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.plot(blue_freqs, blue_power, label="Blue Noise", color="blue")
ax4.plot(gaussian_freqs, gaussian_power, label="Gaussian Noise", color="orange")
ax4.set_title("Log-Log Power Spectrum")
ax4.set_xlabel("Frequency")
ax4.set_ylabel("Power")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()