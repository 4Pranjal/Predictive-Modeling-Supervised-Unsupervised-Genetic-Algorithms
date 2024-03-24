# Traveling Salesman Problem with Genetic Algorithms README

## Overview

This repository contains Python code for solving the Traveling Salesman Problem (TSP) using genetic algorithms (GA). The objective is to find the shortest possible route that visits each city exactly once and returns to the origin city.

## Objective

The objective of this project is to implement a genetic algorithm to find the optimal solution to the TSP. The input is a list of cities and their connections, and the output is a route that minimizes the total traveling time.

## Genetic Algorithm

Given the network of cities G, any vector S may be seen as the chromosome corresponding to the route that travels across all the nodes of G. The objective is the implementation of a genetic algorithm to obtain the vector that minimizes the total time.

## Data and Libraries

To simplify the computation of distances and management of the graphs, we use the TSPLIB python library. This library works with a specific file format that describes different TSP problems. Example datasets can be found on various websites, including [this one](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html).

## Deliverables

### GitHub Repository

The code for this activity is accessible in the following GitHub repository:

[Machine_Learning_Project](https://github.com/4Pranjal/Machine_Learning_Project)

### Description of the Chromosome and Adaptations

Explain in detail how the TSP problem is translated into the chromosome. Describe the techniques implemented for selection, mutation, and crossover. At least two different techniques should be implemented for each. Additionally, explain the choice of population size and how the system reaches a stationary state.

### Results

For 5 problems of different sizes:

1. Describe the dataset, including the URL where the data was collected and an explanation of the data.
2. Present the results obtained with at least 6 different combinations of parameters.
3. For the best solution, include a figure showing the evolution of the minimum total traveling distance in each iteration of the algorithm.

## Instructions

To implement the code, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/4Pranjal/Machine_Learning_Project.git
   ```

2. Install the required libraries:

   ```bash
   pip install tsplib95
   ```

3. Navigate to the directory:

   ```bash
   cd Machine_Learning_Project
   ```

4. Run the Python scripts provided in the repository for solving the TSP using genetic algorithms.

## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
