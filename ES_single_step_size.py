import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    Computes the Ackley function for an n-dimensional input vector x.
    Lower is better (global min ~ 0 near x = 0).
    """
    n = len(x)
    sum_sq = np.sum(x ** 2)         # Sum of squares
    sum_cos = np.sum(np.cos(c * x)) # Sum of cosines

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.exp(1)

def plot_ackley(trajectory=None):
    """
    3D-plot of the Ackley function plus an optional trajectory.
    Only meaningful for dim=2.
    """
    x = np.linspace(-32, 32, 80)
    y = np.linspace(-32, 32, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [ackley(np.array([xi, yi])) for xi, yi in zip(x_row, y_row)]
        for x_row, y_row in zip(X, Y)
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet')

    # Plot the trajectory of the best solution (dim=2)
    if trajectory is not None and len(trajectory) > 0:
        # trajectory is a list of (best_solution, best_fitness) over time
        # best_solution is e.g. np.array([x_val, y_val]) in 2D
        points = np.array([t[0] for t in trajectory])
        best_vals = np.array([ackley(p) for p in points])
        ax.plot(points[:, 0], points[:, 1], best_vals, 'go-', label="ES trajectory")

    plt.title("Ackley Function Landscape")
    plt.show()

def initialize_population(mu, dim, bounds, sigma_init):
    """
    Initialize the parent population and mutation step sizes.
    Each individual has `dim` dimensions, plus one sigma for the entire vector.
    """
    # population: shape (mu, dim)
    population = np.random.uniform(bounds[0], bounds[1], (mu, dim))
    # sigma: shape (mu,) => one step-size per individual
    sigma = np.full((mu,), sigma_init)
    return population, sigma

def mutate(parent, parent_sigma, dim, bounds, lr, sigma_bound):
    """
    Generate a mutated child using Gaussian perturbation and single-sigma self-adaptation.
    
    parent_sigma is a single scalar for the entire chromosome `parent`.
    """
    # Single log-normal update for the step size
    # (one random sample => one step size per individual)
    child_sigma = parent_sigma * np.exp(lr * np.random.normal(0, 1))
    # Clip sigma to ensure it's above sigma_bound
    child_sigma = max(child_sigma, sigma_bound)

    # Mutate the child
    # Here child_sigma is a scalar, so we scale a (dim,)-sized normal
    child = parent + np.random.normal(0, child_sigma, size=dim)
    # Clip the child to the specified bounds
    child = np.clip(child, bounds[0], bounds[1])

    return child, child_sigma

def generate_offspring(population, sigma, mu, lambd, dim, bounds, lr, sigma_bound, fitness_fn):
    """
    Generate offspring from the current population of size `mu`,
    producing `lambd` children in total.
    """
    offspring = []
    for _ in range(lambd):
        parent_idx = np.random.randint(mu)  # random parent selection
        parent = population[parent_idx]
        parent_sigma = sigma[parent_idx]

        child, child_sigma = mutate(
            parent, parent_sigma, dim, bounds, lr, sigma_bound
        )
        fitness = fitness_fn(child)
        offspring.append((child, child_sigma, fitness))
    return offspring

def select_survivors(
    offspring, 
    mu, 
    population, 
    sigma, 
    parent_fitnesses, 
    strategy='comma'
):
    """
    Select the top mu individuals from either:
     - The offspring only (comma strategy), or
     - The union of parents and offspring (plus strategy).

    Returns updated (population, sigma, parent_fitnesses, best_solution, best_fitness).
    """
    if strategy == 'comma':
        # Sort offspring by fitness; best => lowest
        offspring.sort(key=lambda x: x[2])
        selected = offspring[:mu]
        new_population = np.array([ind[0] for ind in selected])
        new_sigma = np.array([ind[1] for ind in selected])
        new_fitnesses = np.array([ind[2] for ind in selected])
        
        best_solution = new_population[0]
        best_fitness = new_fitnesses[0]
        return new_population, new_sigma, new_fitnesses, best_solution, best_fitness

    elif strategy == 'plus':
        # Combine parent + offspring
        off_children = np.array([ind[0] for ind in offspring])
        off_sigmas  = np.array([ind[1] for ind in offspring])
        off_fitnesses = np.array([ind[2] for ind in offspring])

        combined_pop = np.vstack((population, off_children))
        combined_sigma = np.concatenate((sigma, off_sigmas))
        combined_fitness = np.concatenate((parent_fitnesses, off_fitnesses))

        # Sort combined by fitness
        sort_indices = np.argsort(combined_fitness)
        combined_pop = combined_pop[sort_indices]
        combined_sigma = combined_sigma[sort_indices]
        combined_fitness = combined_fitness[sort_indices]

        # Take top mu
        new_population = combined_pop[:mu]
        new_sigma = combined_sigma[:mu]
        new_fitnesses = combined_fitness[:mu]

        best_solution = new_population[0]
        best_fitness = new_fitnesses[0]
        return new_population, new_sigma, new_fitnesses, best_solution, best_fitness

def evolution_strategy(
    fitness_fn,
    dim=2,
    mu=30,
    lambd=200,
    sigma_init=3.0,
    bounds=(-30, 30),
    sigma_bound=0.01,
    max_evals=100000,
    strategy='comma'
):
    """
    Runs an Evolution Strategy (ES) with single step-size self-adaptation to
    minimize a given fitness function (Ackley by default).

    :param fitness_fn: The function to minimize (e.g., ackley).
    :param dim: Dimensionality of the search space.
    :param mu: Number of parents.
    :param lambd: Number of offspring.
    :param sigma_init: Initial step size.
    :param bounds: Tuple (lower_bound, upper_bound) for each dimension.
    :param sigma_bound: Minimum value for sigma (step size).
    :param max_evals: Maximum number of function evaluations.
    :param strategy: 'comma' or 'plus'.
    :return: best_solution, best_fitness, trajectory
    """
    # 1) Initialize population
    population, sigma = initialize_population(mu, dim, bounds, sigma_init)
    # Evaluate parents initially
    parent_fitnesses = np.array([fitness_fn(population[i]) for i in range(mu)])

    # 2) Track the best solution
    best_idx = np.argmin(parent_fitnesses)
    best_solution = population[best_idx]
    best_fitness = parent_fitnesses[best_idx]
    trajectory = [(best_solution, best_fitness)]

    lr = 1 / np.sqrt(dim)  # typical 1/sqrt(dim)
    eval_count = mu  # we already did mu evaluations

    # 3) Main loop
    while eval_count < max_evals:
        # (a) Generate offspring
        offspring = generate_offspring(
            population, sigma, mu, lambd, dim, bounds, lr, sigma_bound, fitness_fn
        )
        eval_count += lambd

        # (b) Select survivors
        population, sigma, parent_fitnesses, new_best_solution, new_best_fitness = select_survivors(
            offspring, mu, population, sigma, parent_fitnesses, strategy=strategy
        )

        # (c) Update global best
        if new_best_fitness < best_fitness:
            best_solution, best_fitness = new_best_solution, new_best_fitness
            trajectory.append((best_solution, best_fitness))

        # Print status periodically
        if eval_count % 1000 == 0:
            print(f"Evaluations: {eval_count}, Best Fitness: {best_fitness:.6f}, Solution: {best_solution}")

    return best_solution, best_fitness, trajectory

def main():
    # Run the ES on Ackley (dim=2)
    best_sol, best_fit, trajectory = evolution_strategy(
        ackley, dim=2, mu=30, lambd=200, sigma_init=3.0, 
        bounds=(-30, 30), sigma_bound=0.01, max_evals=100000,
        strategy='comma'   # or 'plus'
    )
    print(f"Best solution found: {best_sol}")
    print(f"Best fitness: {best_fit}")

    # Plot the 2D Ackley function plus the ES trajectory
    plot_ackley(trajectory)

if __name__ == '__main__':
    main()
