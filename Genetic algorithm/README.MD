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

## Results of Genetic Algorithm Details for TSP

| Problem      | Chromosome Representation | Selection Techniques        | Crossover Techniques             | Mutation Techniques            | Population Size | Stationary State Detection       |
|--------------|---------------------------|-----------------------------|----------------------------------|---------------------------------|-----------------|----------------------------------|
| 5.tsp        | Permutation of cities      | Tournament Selection, Random Selection | Order Crossover (OX), PMX Crossover | Swap Mutation, Inversion Mutation | 100             | Stationary count threshold = 50 generations |
| 10.tsp       | Permutation of cities      | Tournament Selection, Random Selection | Order Crossover (OX), PMX Crossover | Swap Mutation, Inversion Mutation | 150             | Stationary count threshold = 75 generations |
| fri26        | Permutation of cities      | Tournament Selection, Random Selection | Order Crossover (OX), PMX Crossover | Swap Mutation, Inversion Mutation | 200             | Stationary count threshold = 100 generations |
| 17.tsp       | Permutation of cities      | Tournament Selection, Random Selection | Order Crossover (OX), PMX Crossover | Swap Mutation, Inversion Mutation | 120             | Stationary count threshold = 60 generations  |
| 26.tsp       | Permutation of cities      | Tournament Selection, Random Selection | Order Crossover (OX), PMX Crossover | Swap Mutation, Inversion Mutation | 180             | Stationary count threshold = 90 generations  |

![Alt text](data:https://github.com/4Pranjal/Predictive-Modeling-Supervised-Unsupervised-Genetic-Algorithms/blob/main/Genetic%20algorithm/Results/Result_17_dataset.png)

## Conclusion

The analysis of the datasets (5.tsp, 10.tsp, fri26, 17.tsp, and 26.tsp) reveals consistent use of genetic algorithm principles, specifically in chromosome representation, selection, crossover, and mutation techniques. Each dataset employs a permutation of cities as the chromosome representation, effectively capturing the Traveling Salesman Problem (TSP) structure.

1. **Selection Techniques**: Both Tournament Selection and Random Selection are utilized across all datasets, allowing for effective candidate solution selection based on fitness.

2. **Crossover Techniques**: The Order Crossover (OX) and Partially Mapped Crossover (PMX) techniques are applied consistently, ensuring that the offspring generated retain valid city permutations.

3. **Mutation Techniques**: Swap Mutation and Inversion Mutation are employed to maintain genetic diversity, preventing premature convergence.

4. **Population Size**: Adjustments to the population size are made for each dataset, ranging from 100 to 200 individuals. Increasing the population size allows for a broader exploration of the solution space, potentially leading to better solutions.

5. **Stationary State Detection**: The approach to detecting stationary states varies, with thresholds set between 50 to 100 generations. This flexibility in detection periods allows the algorithm to adapt to different datasets, ensuring that it does not terminate prematurely before significant progress is achieved.

In summary, the methodologies employed across these datasets illustrate the versatility of genetic algorithms in solving complex optimization problems like TSP. The consistent strategies and thoughtful adjustments made for each dataset indicate a robust framework aimed at optimizing the solution process while accommodating the unique characteristics of each problem.


## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
