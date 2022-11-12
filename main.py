import random
from typing import List, Optional
import tsplib95
import matplotlib.pyplot as plt
from termcolor import colored

tsp_instance: Optional[tsplib95.models.Problem] = None

# population size
POPULATION_SIZE = 1000

PROBLEM_1_FILEPATH = ['berlin52.tsp', 'berlin52.opt.tour']
PROBLEM_2_FILEPATH = ['a280.tsp', 'a280.opt.tour']
ACTIV_PROBLEM_FILEPATH = []
ALWAYS_START_AT_1 = False


def read_tsp_file(file: str):
    """
    read tsp test instance from tsplib
    """
    global tsp_instance
    tsp_instance = tsplib95.load(file)


class Individual():
    # initialize
    def __init__(self, fitness: int, path: list[int]) -> None:
        # fitness = distance between cities
        self.fitness = fitness
        # path = list of the cities which have been/can be visited
        self.path = path


def get_cities_from_instance():
    return list(tsp_instance.get_nodes())


def get_distance_from_instance(city1: int, city2: int):
    return tsp_instance.get_weight(city1, city2)


def get_fitness(path: list[int]):
    """

    :param path: the list of citites to visit
    :param tsp_instance: tsp test instance
    :return: sum of the distances = fitness
    """

    fitness = 0
    for i in range(len(path) - 1):
        fitness += get_distance_from_instance(path[i], path[i + 1])
    fitness += get_distance_from_instance(path[-1], path[0])
    return fitness


def get_random_path(cities):
    """
    choose a random order of cities
    :return: list of cities
    """
    random.shuffle(cities)
    return cities


def create_initial_population(cities) -> List[Individual]:
    """
    population = list of individuals = list of solutions
    :param1 = list of cities to visit
    :param2 = the test instance in use
    :return: list of individuals = solutions
    """
    population = []

    for i in range(POPULATION_SIZE):
        # need to copy the list of cities because otherwise paths get overwritten
        path = get_random_path(cities.copy())
        fitness = get_fitness(path)
        population.append(Individual(fitness, path))
    return population


def evolutionary_algortihm(population: List[Individual], n_iterations: int, cities, mu, llambda, mutation_probability):
    """
    loop for many generations
    """
    for i in range(n_iterations):
        print()
        print("Generation ", i + 1)
        population = plus_selection(mu, llambda, population, mutation_probability, cities)
        print(f"Fitness of best individuum: {population[0].fitness}")
        if ((i + 1) % 10) == 0 or i == 0:
            plot_tsp(population[0].path, i + 1, population[0].fitness)


def mutation(individual: Individual, mutation_probability: float) -> Individual:
    """

    :param individual: individual that will mutate
    :return: individual that has mutated
    """
    path = individual.path.copy()
    # print("Path to mutate: ", path, "Fitness: ", individual.fitness)
    temp = 0
    # number between 0 and 1
    start = 0
    if ALWAYS_START_AT_1:
        start = 1
    for i in range(start, len(path) - 1):
        random_number = random.random()
        if random_number < mutation_probability:
            # the city with a random number < mutation_probability is being replaced
            # with a random city from the same path so that
            # it is ensured that there is no double occurance of the same city
            temp = path[i]
            new_city = random.choice(path)
            if ALWAYS_START_AT_1:
                while new_city == 1:
                    new_city = random.choice(path)
            index_new_city = path.index(new_city)
            path[i] = new_city
            path[index_new_city] = temp
    # save the new path in the individual
    # ändert sich der Fitness von alleine?
    # print("Mutated path: ", path, "Fitness: ", individual.fitness)
    return Individual(get_fitness(path), path)


def convert_to_valid_path(path_child: List[int], cities: List[int]):
    valid_path = []
    forgotten_cities = []
    """
    # all cities from the build child path but only once
    for city in path_child:
        if city not in valid_path:
            valid_path.append(city)
    # all cities that are missing in the path
    for city in cities:
        if city not in valid_path:
            valid_path.append(city)
    """
    for city in path_child:
        if city not in valid_path:
            valid_path.append(city)
        else:
            valid_path.append(None)

    for city in cities:
        if city not in valid_path:
            forgotten_cities.append(city)

    random.shuffle(forgotten_cities)

    for i, city in enumerate(valid_path):
        if city is None:
            valid_path[i] = forgotten_cities.pop()

    return valid_path


def crossover(individual1: Individual, individual2: Individual, crossovers: int, cities, random_crossovers=True) -> \
        List[Individual]:
    len_path = len(individual1.path)
    individuals_path = [individual1.path.copy(), individual2.path.copy()]
    crossover_points = []

    if random_crossovers:
        for i in range(crossovers):
            crossover_points.append(random.randint(0, len_path))
        crossover_points.sort()
    else:
        step = int(len_path / crossovers)
        for i in range(crossovers):
            crossover_points.append(step)
            step += step

    crossover_points.append(len_path)

    child1_path = []
    child2_path = []
    flip_flop = 0
    old_c = 0
    for c in crossover_points:
        child1_path.extend(individuals_path[flip_flop % 2][old_c:c])
        child2_path.extend(individuals_path[(flip_flop + 1) % 2][old_c:c])
        flip_flop += 1
        old_c = c

    child1_path = convert_to_valid_path(child1_path, cities)
    child2_path = convert_to_valid_path(child2_path, cities)

    child1 = Individual(get_fitness(child1_path), child1_path)
    child2 = Individual(get_fitness(child2_path), child2_path)


    return [child1, child2]


def recombination(individual1: Individual, individual2: Individual, cross_point: int, cities) -> List[Individual]:
    """

    :param individual1: parent one
    :param individual2:  parent two
    :return: recombinated child
    """

    print("I1 path:", individual1.path)
    print("I2 path: ", individual2.path)
    # subset the paths in two with the crossover point
    individual1_subpath_first_half = individual1.path[:cross_point]
    individual1_subpath_second_half = individual1.path[cross_point:]
    individual2_subpath_first_half = individual2.path[:cross_point]
    individual2_subpath_second_half = individual2.path[cross_point:]

    # build the children paths
    path_child1 = individual1_subpath_first_half + individual2_subpath_second_half
    path_child2 = individual2_subpath_first_half + individual1_subpath_second_half
    # recombined paths should only contain each city once
    path_child1 = convert_to_valid_path(path_child1, cities)
    path_child2 = convert_to_valid_path(path_child2, cities)

    fitness_child1 = get_fitness(path_child1)
    fitness_child2 = get_fitness(path_child2)
    # print the offspring :D
    print("child1 path:", path_child1, "Fitness: ", fitness_child1)
    print("child2 path: ", path_child2, "Fitness: ", fitness_child2)

    child1 = Individual(fitness_child1, path_child1)
    child2 = Individual(fitness_child2, path_child2)

    return [child1, child2]


def update_population_with_new_children(population, children):
    population.extend(children)


def plus_selection(mu: int, llambda: int, population: List[Individual], mutation_probability, cities) -> List[
    Individual]:
    """
    1. choose best mu parents
    2. create lambda children from the mu best parents
    3. choose mu best individuals from parents + children

    :param mu: number of individuals for the new population
    :param llambda: number of children to create of the old population
    :param population: old population
    :return: new population with mu indivduals
    """

    # 1. choose best mu parents
    # sort ascending

    best_parents = sorted(population, key=lambda individuum: individuum.fitness)
    # fitness = distance --> low fitness better
    # choose the best mu parents
    best_parents = best_parents[:mu]

    fitness_of_parents = "fitness of Parents: "
    for p in best_parents:
        fitness_of_parents += str(p.fitness) + " "
    print(fitness_of_parents)

    # show the best parents choice
    print("best_parents:")
    print_population(best_parents[:5])
    print()

    new_parents = best_parents

    # 2. create lambda children from the mu best parents
    # print("New Offspring")
    new_offspring = []
    while len(new_offspring) <= llambda:
        random_number = random.random()
        # if < 0.5 -> mutation, if >0.5 -> recombination
        if random_number < 0.5:
            # print("Mutation")
            mutated_ind = mutation(best_parents[random.randint(0, len(best_parents) - 1)], mutation_probability)
            new_offspring.append(mutated_ind)
        else:
            # print("Recombination")
            """
            cross_point_random = random.randint(0, len(population))
            recombinated_ind = recombination(population[random.randint(0, len(population) - 1)],
                                             population[random.randint(0, len(population) - 1)], cross_point_random,
                                             cities)
            new_offspring.extend(recombinated_ind)
            """
            parent_1 = best_parents[random.randint(0, len(best_parents) - 1)]
            parent_2 = best_parents[random.randint(0, len(best_parents) - 1)]
            while parent_1.path == parent_2.path:
                parent_1 = best_parents[random.randint(0, len(best_parents) - 1)]

            recombinated_ind = crossover(parent_1,
                                         parent_2,
                                         6,
                                         cities)

            new_offspring.extend(recombinated_ind)

    # 3. choose mu best individuals from parents + children
    mu_lambda_together = new_offspring + best_parents

    remove_same_path(mu_lambda_together)
    # sort ascending based on the fitness. 
    mu_best_individuals = sorted(mu_lambda_together, key=lambda individuum: individuum.fitness)
    # choose the best mu - individuals
    # mu_best_individuals = mu_best_individuals[:mu]
    return mu_best_individuals


def remove_same_path(population: List[Individual]):
    all_path: List[List[int]] = []
    double_path_at: List[int] = []

    for individual in population.copy():
        for path in all_path:
            if path == individual.path:
                if individual in population:
                    population.remove(individual)
                    break
        all_path.append(individual.path)


def print_population(population):
    for i in range(len(population)):
        print(population[i].path, population[i].fitness)
    print()


def plot_tsp(path: List[int], generation: int, fitness: int):
    x_coords = []
    y_coords = []
    for city in path:
        x_coords.append(tsp_instance.node_coords[city][0])
        y_coords.append(tsp_instance.node_coords[city][1])
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    plt.plot(x_coords, y_coords)
    plt.ylabel('y')
    plt.xlabel('x')
    title = 'Best from generation: ' + str(generation) + ", fitness: " + str(fitness)
    plt.title(title)
    plt.show()


def main(n_iteration, best_parents, n_children, mutation_probability):
    # tsp file load
    file = ACTIV_PROBLEM_FILEPATH[0]
    # 2. Testinstanz hinzufügen TODO()
    read_tsp_file(file)
    # test if tsp can be read
    cities = get_cities_from_instance()
    distance_between_city1_city2 = get_distance_from_instance(cities[0], cities[1])
    # print("Cities: ",cities)
    # print("Distance between first and second city: ", distance_between_city1_city2)
    #
    # create population
    population = create_initial_population(cities)
    if ALWAYS_START_AT_1:
        for individual in population:
            individual.path[0] = 1
            individual.path = convert_to_valid_path(individual.path, cities)
    # print("\n Initial Population: \nPATH	 FITNESS VALUE\n")
    # print_population(population)
    #
    # # mutation
    # mutation(population[0], 0.2)
    #
    # # recombination
    # offspring = recombination(population[1],population[2], 3, tsp_test_instance,cities)
    # # update the population with the new children
    # update_population_with_new_children(population,offspring)
    # print("Population after recombination")
    # # still not 12
    # print_population(population)

    # print("Selektion")
    # best 3 parents and 6 new children
    # plus_selection(3,6,population,0.2, cities, tsp_test_instance)
    # for i in range(len(res)):
    #     print(res[i].path, res[i].fitness)

    print("The Evolutionary Process beginns:")
    # Welchen Einfl uss haben die Parameter auf die Ergebnisse?
    # mit mehr generationen bekommen wir besseren fitness da wir immer die besten auswählen
    evolutionary_algortihm(population, n_iteration, cities, best_parents, n_children, mutation_probability)
    opt = tsplib95.load(ACTIV_PROBLEM_FILEPATH[1])
    print()
    print('optimal tour: ', opt.tours)
    print('fitness of optimal tour: ', tsp_instance.trace_tours(opt.tours))
    # print(get_fitness(opt.tours[0]))


if __name__ == "__main__":
    # immer mit den selben random Zahlen
    # random.seed(41)
    # main(10,3,4,0.2)
    # more generations -> better fitness
    # main(10,3,4,0.2)
    # more best parents ->
    ACTIV_PROBLEM_FILEPATH = PROBLEM_1_FILEPATH
    main(100, 550, 2000, 0.3)
