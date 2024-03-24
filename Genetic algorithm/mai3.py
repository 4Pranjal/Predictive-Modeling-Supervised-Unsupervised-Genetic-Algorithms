# Step 1: Install the required library
# pip install tsplib95
# Step 2: Import necessary libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import tsplib95

# Step 3: Load TSP data
def load_tsp_data(file_path):
    problem = tsplib95.load(file_path)
    return problem

# Step 4: Chromosome representation
# Let's represent a solution as a permutation of cities. 
# We'll use the indices as the order in which the cities are visited.
def initialize_population(population_size, num_cities):
    return [random.sample(range(1, num_cities + 1), num_cities) for _ in range(population_size)]

# Step 5: Fitness function
# Calculate the total distance for a given solution.
def calculate_total_distance(solution, problem):
    distance = 0
    for i in range(len(solution) - 1):
        distance += problem.get_weight(solution[i], solution[i + 1])
    distance += problem.get_weight(solution[-1], solution[0])  # Return to the starting city
    return distance

# Step 6: Selection
# Implement two selection techniques. Here, let's use tournament selection and roulette wheel selection.
def tournament_selection(population, k, problem):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, k)
        best_solution = min(tournament, key=lambda x: calculate_total_distance(x, problem))
        selected_parents.append(best_solution)
    return selected_parents

def roulette_wheel_selection(population, problem):
    fitness_values = [1 / calculate_total_distance(solution, problem) for solution in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_parents = np.random.choice(population, size=len(population), p=probabilities, replace=True)
    return selected_parents

# Step 7: Crossover
# Implement two crossover techniques. Here, let's use order crossover and partially matched crossover (PMX).
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[start:end] + [city for city in parent2 if city not in parent1[start:end]]
    child2 = parent2[start:end] + [city for city in parent1 if city not in parent2[start:end]]
    return child1 + child2[start:] + child2[:start], child2 + child1[start:] + child1[:start]

def partially_matched_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    mapping1 = {parent1[i]: parent2[i] for i in range(start, end)}
    mapping2 = {parent2[i]: parent1[i] for i in range(start, end)}
    child1 = [mapping1.get(city, city) for city in parent1]
    child2 = [mapping2.get(city, city) for city in parent2]
    return child1, child2

# Step 8: Mutation
# Implement two mutation techniques. Here, let's use swap mutation and inversion mutation.
def swap_mutation(solution):
    idx1, idx2 = random.sample(range(len(solution)), 2)
    solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

def inversion_mutation(solution):
    start, end = sorted(random.sample(range(len(solution)), 2))
    solution[start:end] = reversed(solution[start:end])
    return solution

# Step 9: Genetic Algorithm
def genetic_algorithm(file_path, population_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.2):
    problem = load_tsp_data(file_path)
    num_cities = problem.dimension

    population = initialize_population(population_size, num_cities)

    best_distances = []

    for generation in range(generations):
        selected_parents = tournament_selection(population, k=5, problem=problem)
        offspring = []

        for i in range(0, len(selected_parents), 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = inversion_mutation(child2)

            offspring.extend([child1, child2])

        # Replace the old population with the new one
        population = offspring

        # Find and store the best solution in each generation
        best_solution = min(population, key=lambda x: calculate_total_distance(x, problem))
        best_distance = calculate_total_distance(best_solution, problem)
        best_distances.append(best_distance)

        # Print progress
        print(f"Genetic Algorithm Progress: {100 * generation / generations:.2f}%")

    # Return the best solution and its distance
    return best_solution, best_distance, best_distances

# Example usage:
file_path = "C:\\Users\\PRJA\\OneDrive - Carlson Rezidor\\Desktop\\A4\\att48.tsp"
best_solution, best_distance, best_distances = genetic_algorithm(file_path)

# Plot the evolution of the minimum total traveling distance
plt.plot(range(len(best_distances)), best_distances)
plt.xlabel("Generation")
plt.ylabel("Total Distance")
plt.title("Evolution of Minimum Total Traveling Distance")
plt.show()

