import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['white', 'black'])

# Domain size in mm
domain_size = 50  # mm

# Fine grid definition for nearly continuous Fourier computation
fine_grid_points = 1001  # fine resolution for continuous function
x_fine = np.linspace(0, domain_size, fine_grid_points)
y_fine = np.linspace(0, domain_size, fine_grid_points)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

# Discrete grid setup
coarse_grid_points = 50 + 1  # 51 points to cover 50mm with 50 1mm blocks

# Series limits
N = 5  # Higher values make the pattern more complex
M = 5

# Natural wavelength of signal
lambda_x = 1
lambda_y = 1

# Function to calculate coefficients
def alpha_nm(n, m):
    return np.cos(n) * np.cos(m)

def beta_nm(n, m):
    return np.sin(n) * np.cos(m)

def gamma_nm(n, m):
    return np.cos(n) * np.sin(m)

def delta_nm(n, m):
    return np.sin(n) * np.sin(m)

# Compute the Fourier series
def fourier_series(X, Y, N, M):
    f = np.zeros_like(X)
    for n in range(N):
        for m in range(M):
            alpha = alpha_nm(n, m)
            beta = beta_nm(n, m)
            gamma = gamma_nm(n, m)
            delta = delta_nm(n, m)
            f += alpha * np.cos(2 * np.pi * lambda_x * n * X) * np.cos(2 * np.pi * lambda_y * m * Y)
            f += beta * np.sin(2 * np.pi * lambda_x * n * X) * np.cos(2 * np.pi * lambda_y * m * Y)
            f += gamma * np.cos(2 * np.pi * lambda_x * n * X) * np.sin(2 * np.pi * lambda_y * m * Y)
            f += delta * np.sin(2 * np.pi * lambda_x * n * X) * np.sin(2 * np.pi * lambda_y * m * Y)
    return f

# Normalization function
def normalize(F, coarse_points):
    block_size = F.shape[0] // (coarse_points - 1)  # Determine block size for averaging
    F_norm = np.zeros((coarse_points - 1, coarse_points - 1))
    for i in range(coarse_points - 1):
        for j in range(coarse_points - 1):
            start_x = i * block_size
            end_x = start_x + block_size
            start_y = j * block_size
            end_y = start_y + block_size
            # Assign 1 if the average of the block is greater than 0
            F_norm[i, j] = 1 if np.mean(F[start_x:end_x, start_y:end_y]) > 0 else 0
    return F_norm

# Evaluate the Fourier series on the fine grid
F = fourier_series(X_fine, Y_fine, N, M)
print("Continuous Fourier Series:")
print(F)

# Compute the normalized Fourier transform with the defined block size
F_norm = normalize(F, coarse_grid_points)
print("Normalized (Discrete) Fourier Series:")
print(F_norm)

# Setup figure and subplots
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# 3D plot on the left
surface = ax1.plot_surface(X_fine, Y_fine, F, cmap='viridis')
fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=5)
ax1.set_title('2D Fourier Series Visualization in 3D Space')

# 2D mesh grid on the right
cax = ax2.imshow(F_norm, extent=(x_fine.min(), x_fine.max(), y_fine.min(), y_fine.max()), origin='lower', cmap=cmap, vmin=0, vmax=1)
fig.colorbar(cax, ax=ax2)
ax2.set_title('Fourier Series Projection')

plt.show()
