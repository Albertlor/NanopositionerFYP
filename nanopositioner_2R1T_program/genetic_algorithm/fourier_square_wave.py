import pygad
import numpy as np
import matplotlib.pyplot as plt
import time

# Generate the square wave
def square_wave(x):
    return np.sign(np.sin(x))

# Define the full Fourier series with constant, sine, and cosine terms
def fourier_series(x, coefficients):
    a_0 = coefficients[0]  # Constant term
    result = a_0 * np.ones_like(x)  # Start with the constant term

    num_terms = (len(coefficients) - 1) // 2  # Number of harmonics (excluding constant term)
    a_n = coefficients[1:num_terms+1]  # Cosine coefficients
    b_n = coefficients[num_terms+1:]  # Sine coefficients

    for n in range(1, num_terms + 1):
        result += a_n[n-1] * np.cos(n * x) + b_n[n-1] * np.sin(n * x)

    return result

# Fitness function based purely on minimizing MSE (no prior assumptions)
def fitness_function(ga_instance, solution, solution_idx):
    x_values = np.linspace(0, 2 * np.pi, 1000)  # Points to evaluate the series
    square_wave_values = square_wave(x_values)  # Target square wave
    
    # Get the Fourier series approximation for the current set of coefficients
    fourier_values = fourier_series(x_values, solution)
    
    # Calculate mean squared error (MSE) between Fourier series and square wave
    mse = np.mean((fourier_values - square_wave_values) ** 2)

    # Since GA maximizes fitness, we return the inverse of the MSE
    return 1.0 / (mse + 1E-6)  # Small epsilon to avoid division by zero

# Configure the genetic algorithm without assuming any prior pattern
ga_instance = pygad.GA(
    num_generations=500,  # More generations to allow for better convergence
    num_parents_mating=2,  # Increase parents for better diversity
    fitness_func=fitness_function,  # Fitness function
    sol_per_pop=100,  # Larger population size for better exploration
    num_genes=11,  # 1 constant term + 5 cosine terms + 5 sine terms
    init_range_low=-10.0,  # Larger range to explore various coefficients
    init_range_high=10.0,
    mutation_percent_genes=1,  # Increase mutation rate to encourage exploration
    mutation_type="random",  # Use random mutation
    crossover_type="single_point",  # Use single-point crossover
    parent_selection_type="tournament",  # Tournament selection for mating
    keep_parents=2,  # Keep top 5 parents for the next generation
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

# Generate values to plot
x_values = np.linspace(0, 2 * np.pi, 1000)
square_wave_values = square_wave(x_values)
fourier_approximation = fourier_series(x_values, solution)

# Display the errors
mse = np.mean((fourier_approximation - square_wave_values) ** 2)
print(f"Mean Squared Error: {mse}")

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot the exact square wave in the first subplot
ax1.plot(x_values, square_wave_values, label='Exact Square Wave', color='blue')
ax1.set_title('Exact Square Wave')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True)

# Plot the Fourier approximation in the second subplot
ax2.plot(x_values, fourier_approximation, label='Fourier Approximation', color='red')
ax2.set_title('Fourier Series Approximation of Square Wave')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

# Plot fitness over generations
ga_instance.plot_fitness()
