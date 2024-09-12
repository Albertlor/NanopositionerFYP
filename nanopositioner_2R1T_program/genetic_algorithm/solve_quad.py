import pygad
import numpy as np
import time

# Coefficients of the quadratic equation ax^2 + bx + c = 0
a = 1
b = 2
c = 5

# Define the fitness function to handle two complex roots
def fitness_function(ga_instance, solution, solution_idx):
    # Extract real and imaginary parts for both roots
    real_part_1 = solution[0]  # Real part of first root
    imag_part_1 = solution[1]  # Imaginary part of first root
    real_part_2 = solution[2]  # Real part of second root
    imag_part_2 = solution[3]  # Imaginary part of second root
    
    # Combine them into two complex numbers (roots)
    root_1 = complex(real_part_1, imag_part_1)
    root_2 = complex(real_part_2, imag_part_2)
    
    # Evaluate the quadratic equation for both roots
    equation_result_1 = a * (root_1 ** 2) + b * root_1 + c
    equation_result_2 = a * (root_2 ** 2) + b * root_2 + c
    
    # Calculate the absolute errors for both roots
    error_1 = np.abs(equation_result_1)
    error_2 = np.abs(equation_result_2)
    
    # The fitness is the inverse of the sum of the errors (minimizing the errors)
    fitness = 1.0 / (error_1 + error_2 + 1E6)  # Small epsilon to avoid division by zero
    
    return fitness

# Track the total time taken for all generations
total_time = 0

# Define the callback function for each generation
def on_generation(ga_instance):
    # This method is called after each generation is completed
    print(f"Generation {ga_instance.generations_completed} completed.")
    pass

# Configure the genetic algorithm
ga_instance = pygad.GA(
    num_generations=100,  # Number of generations (iterations)
    num_parents_mating=2,  # Number of parents that mate to produce offspring
    fitness_func=fitness_function,  # Fitness function
    sol_per_pop=100,  # Population size (number of solutions)
    num_genes=4,  # We now have 4 genes: real and imaginary parts of both roots
    init_range_low=-10,  # Initial range for real and imaginary parts of roots
    init_range_high=10,
    mutation_num_genes=2,  # Ensures at least 2 genes mutate
    mutation_type="random",  # Type of mutation
    crossover_type="single_point",  # Type of crossover
    parent_selection_type="tournament",  # Parent selection method
    keep_parents=2,  # Number of parents to keep for the next generation
    on_generation=on_generation  # Callback for after each generation
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
real_part_1 = solution[0]
imag_part_1 = solution[1]
real_part_2 = solution[2]
imag_part_2 = solution[3]
root_1 = complex(real_part_1, imag_part_1)
root_2 = complex(real_part_2, imag_part_2)

print(f"Best solution (first root): x1 = {root_1}")
print(f"Best solution (second root): x2 = {root_2}")
print(f"Fitness of the solution: {solution_fitness}")

# Error Evaluation: Calculate the value of the equation at the found roots
equation_result_1 = a * (root_1 ** 2) + b * root_1 + c
equation_result_2 = a * (root_2 ** 2) + b * root_2 + c
error_1 = np.abs(equation_result_1)  # Absolute error for root 1
error_2 = np.abs(equation_result_2)  # Absolute error for root 2

# # Display the errors
# print(f"Equation result for first root: {equation_result_1}")
# print(f"Equation result for second root: {equation_result_2}")
# print(f"Absolute error for first root (should be close to 0): {error_1}")
# print(f"Absolute error for second root (should be close to 0): {error_2}")

# # Check if both errors are below a threshold (for justification)
# error_threshold = 0.001  # Define a threshold
# if error_1 < error_threshold and error_2 < error_threshold:
#     print(f"Both solutions are acceptable with errors of {error_1} and {error_2}.")
# else:
#     print(f"The solutions are not acceptable. Errors are {error_1} and {error_2}.")

# Plot fitness over generations
ga_instance.plot_fitness()
