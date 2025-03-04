import numpy as np
import torch
import os
from tqdm import tqdm
import concurrent.futures

NUM_REF_MASKS = 10000         # how many reference masks to generate via simulated annealing
SEQ_LENGTH    = 48            # time dimension
NUM_FEATURES  = 35            # feature dimension
SAVE_MASKS_PATH = "blue_noise_ref_masks.npz"
SAVE_CHOL_PATH  = "blue_noise_chol_matrix.pt"

INIT_TEMP        = 5.0       # initial annealing temperature
FINAL_TEMP       = 0.1
ANNEALING_RATE   = 0.995     # how quickly the temperature drops
NUM_ITERATIONS   = 5000      # iterations per single mask
POWER_EXP        = 2.0       # exponent for "ideal" high frequency weighting
ADAPTIVE_DECAY   = 500       # Iterations before decreasing temp if stuck
NBINS            = 50        # number of bins for radial averaging

def precompute_radial_bins(nrows, ncols, nbins):
    # Create frequency grids for both dimensions
    fx = np.fft.fftfreq(ncols)
    fy = np.fft.fftfreq(nrows)
    fx, fy = np.meshgrid(fx, fy)
    # Shift grids so that the zero frequency is in the center
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftshift(fy)
    # Compute radial frequency for each pixel
    r = np.sqrt(fx**2 + fy**2)
    r_flat = r.flatten()
    # Define bin edges
    r_bins = np.linspace(r_flat.min(), r_flat.max(), nbins + 1)
    bin_centers = np.zeros(nbins)
    for i in range(nbins):
        bin_centers[i] = (r_bins[i] + r_bins[i+1]) / 2
    return r, r_bins, bin_centers

# Precompute constants for energy computation
radial_r, r_bins, bin_centers = precompute_radial_bins(SEQ_LENGTH, NUM_FEATURES, NBINS)

def compute_energy(mask):
    mask_f = mask.astype(float)
    # Compute the 2D FFT and power spectrum
    fft2d = np.fft.fft2(mask_f)
    psd2d = np.abs(fft2d)**2
    # Shift zero frequency to center
    psd2d_shifted = np.fft.fftshift(psd2d)

    # Flatten arrays for binning
    r_flat = radial_r.flatten()
    psd_flat = psd2d_shifted.flatten()
    
    # Radial average
    radial_profile = np.zeros(NBINS)
    for i in range(NBINS):
        bin_mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.any(bin_mask):
            radial_profile[i] = np.mean(psd_flat[bin_mask])
        else:
            radial_profile[i] = 0.0

    def frequency_penalty(f):
        return np.abs(f)**POWER_EXP + 0.1 / (np.abs(f) + 1e-4)

    ideal_spectrum = frequency_penalty(bin_centers)
    ideal_spectrum[0] = 0.0  # Remove the DC component

    # Normalize both spectra
    ideal_spectrum /= np.sum(ideal_spectrum)
    if np.sum(radial_profile) > 0:
        radial_profile /= np.sum(radial_profile)
    
    # Mean squared error between the actual and ideal spectrum
    energy = np.mean((radial_profile - ideal_spectrum)**2)
    return energy

def anneal_one_mask(seq_len=SEQ_LENGTH, n_feat=NUM_FEATURES):
    total_elements = seq_len * n_feat
    # Create a mask with fixed density (50% ones, 50% zeros)
    num_ones = total_elements // 2
    mask_flat = np.zeros(total_elements, dtype=np.int32)
    mask_flat[:num_ones] = 1
    np.random.shuffle(mask_flat)
    mask = mask_flat.reshape(seq_len, n_feat)
    
    energy = compute_energy(mask)
    temperature = INIT_TEMP
    acceptance_rate = 0

    for step in range(NUM_ITERATIONS):
        num_swaps = 1
        
        # Get indices for ones and zeros
        ones_idx = np.argwhere(mask == 1)
        zeros_idx = np.argwhere(mask == 0)
        if len(ones_idx) == 0 or len(zeros_idx) == 0:
            break
        
        chosen_ones = ones_idx[np.random.choice(len(ones_idx), size=num_swaps, replace=False)]
        chosen_zeros = zeros_idx[np.random.choice(len(zeros_idx), size=num_swaps, replace=False)]
        
        # Perform the swaps
        for i in range(num_swaps):
            r1, c1 = chosen_ones[i]
            r2, c2 = chosen_zeros[i]
            mask[r1, c1], mask[r2, c2] = mask[r2, c2], mask[r1, c1]
        
        new_energy = compute_energy(mask)
        deltaE = new_energy - energy

        if (deltaE < 0) or (np.exp(-deltaE / temperature) > np.random.rand()):
            energy = new_energy
            acceptance_rate += 1
        else:
            # Revert the swaps if not accepted
            for i in range(num_swaps):
                r1, c1 = chosen_ones[i]
                r2, c2 = chosen_zeros[i]
                mask[r1, c1], mask[r2, c2] = mask[r2, c2], mask[r1, c1]
        
        if step % ADAPTIVE_DECAY == 0 and acceptance_rate < 0.01 * ADAPTIVE_DECAY:
            temperature *= ANNEALING_RATE * 1.2
            acceptance_rate = 0
        else:
            temperature *= ANNEALING_RATE

        if temperature < FINAL_TEMP:
            break

    return mask

def build_blue_noise_cov():
    if not os.path.exists(SAVE_MASKS_PATH):
        # Dont melt your computer
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(anneal_one_mask) for _ in range(NUM_REF_MASKS)]
            all_masks = []
            for fut in tqdm(concurrent.futures.as_completed(futures), total=NUM_REF_MASKS, desc="Annealing BN masks"):
                try:
                    all_masks.append(fut.result())
                except Exception as exc:
                    print(f"An error occurred: {exc}")
        
        all_masks = np.stack(all_masks, axis=0)
        np.savez_compressed(SAVE_MASKS_PATH, masks=all_masks)
    else:
        loaded = np.load(SAVE_MASKS_PATH)
        all_masks = loaded["masks"]
        print(f"[INFO] Loaded {all_masks.shape[0]} reference BN masks from {SAVE_MASKS_PATH}")

    N = all_masks.shape[0]
    flat_len = SEQ_LENGTH * NUM_FEATURES
    data_flat = all_masks.reshape(N, flat_len).astype(np.float32)
    mean_flat = np.mean(data_flat, axis=0, keepdims=True)

    cov_mat = np.zeros((flat_len, flat_len), dtype=np.float64)
    for i in tqdm(range(N), desc="Computing Covariance Matrix"):
        diff = data_flat[i] - mean_flat
        cov_mat += np.outer(diff, diff)
    cov_mat /= (N - 1)

    return cov_mat.astype(np.float32)

def nearest_spd(A, num_iters=5):
    A_sym = 0.5 * (A + A.T)
    for _ in range(num_iters):
        w, v = np.linalg.eigh(A_sym)
        w_clamped = np.maximum(w, 1e-7)
        A_sym = (v * w_clamped) @ v.T
    return A_sym

def compute_and_save_chol():
    print("[INFO] Computing blue noise covariance matrix...")
    cov_mat = build_blue_noise_cov()

    print("[INFO] Converting to nearest SPD matrix...")
    stable_cov = nearest_spd(cov_mat, num_iters=3)
    stable_cov += 1e-7 * np.eye(stable_cov.shape[0], dtype=stable_cov.dtype)

    try:
        with tqdm(total=1, desc="Computing Cholesky Decomposition") as pbar:
            L_chol = np.linalg.cholesky(stable_cov)
            pbar.update(1)
        L_torch = torch.from_numpy(L_chol).float()
        torch.save(L_torch, SAVE_CHOL_PATH)
        print(f"[INFO] Saved Cholesky factor to {SAVE_CHOL_PATH}")
    except np.linalg.LinAlgError:
        print("[ERROR] Cholesky decomposition failed. Increase diagonal adjustment or refine SPD approach.")

if __name__ == "__main__":
    compute_and_save_chol()