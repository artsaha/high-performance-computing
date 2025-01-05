from mpi4py import MPI
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

def random_walk_transfer(grid, local_grid, steps):
    for _ in range(steps):
        new_local_grid = local_grid.copy()
        for i in range(local_grid.shape[0]):
            for j in range(local_grid.shape[1]):
                if local_grid[i, j] > 0:
                    di, dj = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    ni, nj = (i + di) % local_grid.shape[0], (j + dj) % local_grid.shape[1]
                    transfer_amount = local_grid[i, j] * 0.45
                    new_local_grid[i, j] -= transfer_amount
                    new_local_grid[ni, nj] += transfer_amount
        local_grid = new_local_grid

    return local_grid

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid_size = 200
    steps = 400
    heat_source = [(100, 100)]

    if rank == 0:
        grid = initialize_grid(grid_size, heat_source)
    else:
        grid = None

    # Scatter the grid among processes
    local_grid_size = grid_size // size
    local_grid = np.zeros((local_grid_size, grid_size), dtype=float)
    comm.Scatter(grid, local_grid, root=0)

    # Measure execution time and memory usage
    if rank == 0:
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 ** 2

    # Perform Monte Carlo heat transfer simulation
    local_grid = random_walk_transfer(grid, local_grid, steps)

    # Gather the grid from all processes
    comm.Gather(local_grid, grid, root=0)

    if rank == 0:
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

        with open('parallel.txt', 'w') as file:
            messages = [
                f"execution time = {execution_time:.2f} seconds",
                f"memory usage = {memory_usage:.2f} MB",
                f"grid size = {grid_size}",
                f"number of steps for computation = {steps}",
            ]

            file.writelines("% s\n" % data for data in messages)


if __name__ == "__main__":
    main()
