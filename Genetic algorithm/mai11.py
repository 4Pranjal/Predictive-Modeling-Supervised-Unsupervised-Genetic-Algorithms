import random
from itertools import combinations

class TSPProblem:
    def __init__(self, city_coordinates):
        self.city_coordinates = city_coordinates
        self.num_cities = len(city_coordinates)

    def get_weight(self, city1, city2):
        x1, y1 = self.city_coordinates[city1]
        x2, y2 = self.city_coordinates[city2]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def calculate_total_distance(route, problem):
    total_distance = 0
    for i in range(len(route)):
        total_distance += problem.get_weight(route[i], route[(i + 1) % len(route)])
    return total_distance

def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    for i in range(start, end + 1):
        child[i] = parent1[i]
    remaining_elements = [elem for elem in parent2 if elem not in child]
    remaining_index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining_elements[remaining_index]
            remaining_index += 1
    return child

def genetic_algorithm(problem, population_size, generations, crossover_rate, mutation_rate, tournament_size):
    population = [random.sample(range(problem.num_cities), problem.num_cities) for _ in range(population_size)]
    for generation in range(generations):
        population = sorted(population, key=lambda x: calculate_total_distance(x, problem))
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population

    best_solution = min(population, key=lambda x: calculate_total_distance(x, problem))
    best_distance = calculate_total_distance(best_solution, problem)
    return best_solution, best_distance

def tournament_selection(population, size):
    tournament = random.sample(population, size)
    return min(tournament, key=lambda x: calculate_total_distance(x, problem))

def mutate(route):
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def run_algorithm_with_parameters(problem, population_size, generations, crossover_rate, mutation_rate, tournament_size):
    best_solution, best_distance = genetic_algorithm(problem, population_size, generations, crossover_rate, mutation_rate, tournament_size)
    print(f"Best solution: {best_solution}")
    print(f"Best distance: {best_distance}")

# Coordinates for the 17 cities in the gr17 dataset
city_coordinates = [
    (20833.3333, 17100.0000),
    (20900.0000, 17066.6667),
    (21300.0000, 13016.6667),
    (21600.0000, 14150.0000),
    (21600.0000, 14966.6667),
    (21600.0000, 16500.0000),
    (22183.3333, 13133.3333),
    (22583.3333, 14300.0000),
    (22683.3333, 12716.6667),
    (23616.6667, 15866.6667),
    (23700.0000, 15933.3333),
    (23883.3333, 14533.3333),
    (24166.6667, 13250.0000),
    (25149.1667, 12365.8333),
    (26133.3333, 14500.0000),
    (26150.0000, 10550.0000),
    (26283.3333, 12766.6667)
]

# Create a TSPProblem instance
problem = TSPProblem(city_coordinates)

# Run the genetic algorithm with specified parameters
run_algorithm_with_parameters(problem, population_size=100, generations=1000, crossover_rate=0.8, mutation_rate=0.1, tournament_size=5)
