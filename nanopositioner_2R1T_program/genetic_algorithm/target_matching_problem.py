import pygad
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.colors import ListedColormap

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
N = 2  # Higher values make the pattern more complex
M = 2

epsilon = 1E-30

# Compute the Fourier series
def fourier_series(X, Y, N, M, coefficients, lambda_x=0.01, lambda_y=0.01):
    f = np.zeros_like(X)
    index = 0
    for n in range(N):
        for m in range(M):
            alpha = coefficients[index]
            beta = coefficients[index + 1]
            gamma = coefficients[index + 2]
            delta = coefficients[index + 3]
            index += 4
            f += alpha * np.cos(2 * np.pi * lambda_x * n * X) * np.cos(2 * np.pi * lambda_y * m * Y)
            f += beta * np.sin(2 * np.pi * lambda_x * n * X) * np.cos(2 * np.pi * lambda_y * m * Y)
            f += gamma * np.cos(2 * np.pi * lambda_x * n * X) * np.sin(2 * np.pi * lambda_y * m * Y)
            f += delta * np.sin(2 * np.pi * lambda_x * n * X) * np.sin(2 * np.pi * lambda_y * m * Y)
    return f

def evaluate_conditions(F, solid_positions, void_positions, coarse_points):
    block_size = F.shape[0] // (coarse_points - 1)
    F_norm = np.zeros((coarse_points - 1, coarse_points - 1))
    fitness = 0
    for i in range(coarse_points - 1):
        for j in range(coarse_points - 1):
            start_x = i * block_size
            end_x = start_x + block_size
            start_y = j * block_size
            end_y = start_y + block_size
            block_mean = np.mean(F[start_x:end_x, start_y:end_y])
            F_norm[i, j] = 1 if block_mean > 0 else 0

            # Condition checking based on target
            if (block_mean > 0 and (i, j) in solid_positions) or (block_mean <= 0 and (i, j) in void_positions):
                fitness += 1
            else:
                fitness -= 1

    return F_norm, fitness

def define_target(coarse_points):
    F_norm_target = np.zeros((coarse_points - 1, coarse_points - 1))
    for i in range(coarse_points - 1):
        F_norm_target[i, 0:3] = 1
        F_norm_target[i, coarse_points - 4:] = 1
        for j in range(coarse_points - 1):
            if i == 0:
                F_norm_target[i:3, j] = 1
    return F_norm_target

def define_target_positions(target):
    # Find positions where the target is 1
    solid_positions = []
    void_positions = []
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i, j] == 1:
                solid_positions.append((i, j))
            else:
                void_positions.append((i, j))
    return solid_positions, void_positions

def fitness_function(ga_instance, solution, solution_idx):
    F = fourier_series(X_fine, Y_fine, N, M, solution)
    target = define_target(coarse_grid_points)
    F_norm, _ = evaluate_conditions(F, [], [], coarse_grid_points)

    solid_positions, void_positions = define_target_positions(target)
    n_solid = len(solid_positions)
    n_void = len(void_positions)

    correct_solid = sum(1 for pos in solid_positions if F_norm[pos] == 1)
    correct_void = sum(1 for pos in void_positions if F_norm[pos] == 0)

    # Weight solid positions higher due to their scarcity
    fitness = (correct_solid / n_solid) + (correct_void / n_void)

    return fitness

# Configure the genetic algorithm without assuming any prior pattern
ga_instance = pygad.GA(
    num_generations=100,  # More generations to allow for better convergence
    num_parents_mating=5,  # Increase parents for better diversity
    fitness_func=fitness_function,  # Fitness function
    sol_per_pop=100,  # Larger population size for better exploration
    num_genes=N * M * 4,  # N*M alpha terms, N*M beta terms, N*M gamma terms, N*M delta terms
    init_range_low=-1.0,  # Larger range to explore various coefficients
    init_range_high=1.0,
    mutation_num_genes=1,  # Increase mutation rate to encourage exploration
    mutation_type="random",  # Use random mutation
    crossover_type="single_point",  # Use single-point crossover
    parent_selection_type="tournament",  # Tournament selection for mating
    keep_parents=5,  # Keep top 5 parents for the next generation
)

# Measure the total time for the entire run
start_time = time.time()

# Run the genetic algorithm
ga_instance.run()

# Calculate total time and average time per generation
total_time = time.time() - start_time
average_time_per_generation = total_time / ga_instance.generations_completed
print(f"Average solving time per generation: {average_time_per_generation:.5f} seconds")

# Extract the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best Fourier coefficients: {solution}")
print(f"Fitness of the solution: {solution_fitness}")

# Define the target pattern
F_norm_target = define_target(coarse_grid_points)
print("Target:")
print(F_norm_target)

# Continuous Fourier series approximation using GA
F_best = fourier_series(X_fine, Y_fine, N, M, solution)
print("Continuous Fourier Series:")
print(F_best)

# Normalized (discrete) Fourier series
F_norm_best, _ = evaluate_conditions(F_best, [], [], coarse_grid_points)
print("Normalized (discrete) Fourier Series:")
print(F_norm_best)

# Modified Section Start

# Split F_norm_best and F_norm_target into left and right halves
mid_col = (coarse_grid_points - 1) // 2  # Adjusted for zero-based indexing
F_norm_left = F_norm_best[:, :mid_col]
F_norm_right = F_norm_best[:, mid_col:]
F_norm_target_left = F_norm_target[:, :mid_col]
F_norm_target_right = F_norm_target[:, mid_col:]

# Compute error percentages for each half based on solid elements

# For left half
solid_positions_left = np.argwhere(F_norm_target_left == 1)
total_solid_elements_left = solid_positions_left.shape[0]

# Count mismatches at solid positions
mismatches_left = np.sum(F_norm_left != F_norm_target_left)

error_percentage_left = (mismatches_left / total_solid_elements_left) * 100

# For right half
solid_positions_right = np.argwhere(F_norm_target_right == 1)
total_solid_elements_right = solid_positions_right.shape[0]

# Count mismatches at solid positions
mismatches_right = np.sum(F_norm_right != F_norm_target_right)

error_percentage_right = (mismatches_right / total_solid_elements_right) * 100

print(f"Error Percentage Left Half: {error_percentage_left}%")
print(f"Error Percentage Right Half: {error_percentage_right}%")

# Determine which half is better
if error_percentage_left <= error_percentage_right:
    better_half = 'left'
    F_norm_half = F_norm_left
    last_col_index = F_norm_half.shape[1] - 1
    # # Get rows where the rightmost column has solid elements
    # rows_with_solid = np.where(F_norm_half[:, last_col_index] == 1)[0]
    # # Add a column based on these rows
    # additional_column = np.zeros((F_norm_half.shape[0], 1))
    # additional_column[rows_with_solid, 0] = 1
    # # Append the additional column to F_norm_half
    # F_norm_half_extended = np.concatenate((F_norm_half, additional_column), axis=1)
    # Reflect the half and concatenate
    F_norm_full = np.concatenate((F_norm_half, np.flip(F_norm_half, axis=1)), axis=1)
else:
    better_half = 'right'
    F_norm_half = F_norm_right
    first_col_index = 0
    # # Get rows where the leftmost column has solid elements
    # rows_with_solid = np.where(F_norm_half[:, first_col_index] == 1)[0]
    # # Add a column based on these rows
    # additional_column = np.zeros((F_norm_half.shape[0], 1))
    # additional_column[rows_with_solid, 0] = 1
    # # Prepend the additional column to F_norm_half
    # F_norm_half_extended = np.concatenate((additional_column, F_norm_half), axis=1)
    # Reflect the half and concatenate
    F_norm_full = np.concatenate((np.flip(F_norm_half, axis=1), F_norm_half), axis=1)

print(f"Better Half Chosen: {better_half.capitalize()} Half")

# Calculate error percentage of the final pattern based on solid elements
solid_positions_full = np.argwhere(F_norm_target == 1)
total_solid_elements_full = solid_positions_full.shape[0]

# Count mismatches at solid positions
mismatches_full = np.sum(F_norm_full != F_norm_target)

error_percentage_full = (mismatches_full / total_solid_elements_full) * 100

print(f"Final Error Percentage after Reflection: {error_percentage_full}%")

# Modified Section End

# Setup figure and subplots
fig = plt.figure(figsize=(18, 6))  # Adjusted for three subplots
cmap = ListedColormap(['white', 'black'])

# Subplot 1: Target Pattern (2D)
ax1 = fig.add_subplot(131)
ax1.imshow(F_norm_target, cmap=cmap, extent=(0, domain_size, 0, domain_size), origin='lower')
ax1.set_title("Target Pattern")
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.invert_yaxis()

# Subplot 2: Best Fourier Series (3D)
ax2 = fig.add_subplot(132, projection='3d')
surface = ax2.plot_surface(X_fine, Y_fine, F_best, cmap='viridis')
fig.colorbar(surface, ax=ax2, shrink=0.5, aspect=5)
ax2.set_title('3D Visualization of Best Fourier Series')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Amplitude')

# Subplot 3: Final Pattern after Reflection
ax3 = fig.add_subplot(133)
ax3.imshow(F_norm_full, cmap=cmap, extent=(0, domain_size, 0, domain_size), origin='lower')
ax3.set_title("Final Symmetric Pattern")
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.invert_yaxis()

plt.tight_layout()
plt.show()

# Plot fitness over generations
ga_instance.plot_fitness()
