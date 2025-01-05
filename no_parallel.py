import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import time
import psutil

def initialize_grid(grid_size, heat_source):
    grid = np.zeros((grid_size, grid_size), dtype=float)
    for source in heat_source:
        grid[source] = 100.0  # Set the heat source to a high temperature
    return grid

def random_walk_transfer(grid, steps):
    for _ in range(steps):
        new_grid = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 0:
                    di, dj = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    ni, nj = (i + di) % grid.shape[0], (j + dj) % grid.shape[1]
                    transfer_amount = grid[i, j] * 0.25
                    new_grid[i, j] -= transfer_amount
                    new_grid[ni, nj] += transfer_amount
        grid = new_grid

    return grid

def main():
    grid_size = 100
    steps = 100
    heat_source = [(50, 50)]

    grid = initialize_grid(grid_size, heat_source)

    # Measure execution time and memory usage
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 ** 2

    # Perform Monte Carlo heat transfer simulation
    grid = random_walk_transfer(grid, steps)

    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 ** 2
    execution_time = end_time - start_time
    memory_usage = final_memory - initial_memory

    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Visualize the final grid using seaborn
    sns.heatmap(grid, cmap='rocket', cbar_kws={'label': 'Temperature'})
    plt.title('Heat Distribution')
    plt.show()

    with open('no_parallel.txt', 'w') as file:
        messages = [
            f"execution time = {execution_time:.2f} seconds",
            f"memory usage = {memory_usage:.2f} MB",
            f"grid size = {grid_size}",
            f"number of steps for computation = {steps}",
        ]

        file.writelines("% s\n" % data for data in messages)

if __name__ == "__main__":
    main()
