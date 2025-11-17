import numpy as np
import matplotlib.pyplot as plt
import random as rng
import torch

def make_data():
    def random_polynomial_function(degree=5, n_points=100, seed=None):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 1, n_points)
        
        coeffs = rng.normal(0, 1, degree+1)
        f = np.polyval(coeffs, x)
        
        # squash into (-1, 1) with tanh
        f = np.tanh(f)
        return f


    def random_fourier_function(n_terms=5, n_points=100, seed=None):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 1, n_points)
        
        # random frequencies, amplitudes, and phases
        coeffs = rng.normal(0, 0.3, size=n_terms)  # amplitudes
        freqs = rng.integers(1, 5, size=n_terms)  # frequencies
        phases = rng.uniform(0, 2*np.pi, size=n_terms)  # phases
        
        f = np.zeros_like(x)
        for a, w, phi in zip(coeffs, freqs, phases):
            f += a * np.sin(2*np.pi*w*x + phi)
        
        # normalize into (-1, 1)
        f /= np.max(np.abs(f)) + 1e-8
        return f

    batch_N = 20000
    time_N = 100

    time_grid = np.linspace(0, 1, time_N)
    f_data = np.zeros((batch_N, time_N, 1))
    u_data = np.zeros((batch_N, time_N, 1))

    dt = time_grid[1] - time_grid[0]

    for i in range(batch_N):
        print(i)
        choice = rng.choice(["fourier", "poly"])
        if choice == "fourier":
            f_data[i, :, 0] = random_fourier_function(np.random.randint(1, 10))
        else:
            f_data[i, :, 0] = random_polynomial_function(np.random.randint(1, 6))

        for j in range(1, time_N):
            u_data[i, j, 0] = u_data[i, j-1, 0] + 5.0*dt*f_data[i, j, 0]

    save_path = "/localdata/komso/datasets/DeepONet_data_integrator"

    torch.save(torch.tensor(time_grid.reshape(-1, 1), dtype=torch.float32), f"{save_path}/input_t.pt")
    torch.save(torch.tensor(f_data, dtype=torch.float32), f"{save_path}/input_f.pt")
    torch.save(torch.tensor(u_data, dtype=torch.float32), f"{save_path}/output_u.pt")


def integrator(t_data, f_data):
    u_data = torch.zeros_like(f_data)
    delta_t = t_data[1] - t_data[0]
    for i in range(1, len(t_data)):
        u_data[:, i, 0] = u_data[:, i-1, 0] + delta_t * 5.0*f_data[:, i, 0]
    return u_data