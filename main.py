import random
from typing import List, Optional
import tsplib95
import matplotlib.pyplot as plt
from termcolor import colored

tsp_instance: Optional[tsplib95.models.Problem] = None

# population size
POPULATION_SIZE = 600

ch130_FILEPATH = ['ch130.tsp', 'ch130.opt.tour']
berlin52_FILEPATH = ['berlin52.tsp', 'berlin52.opt.tour']
a280_FILEPATH = ['a280.tsp', 'a280.opt.tour']
ACTIV_PROBLEM_FILEPATH = []
ALWAYS_START_AT_1 = False
WITH_GREEDY = True
PLOT_EVERY_X_GENERATIONS = 20
BESTFINESS = []


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
        print()
        print(f"[Fitness of best individuum for genration {i+1}: {population[0].fitness}]")
        BESTFINESS.append(population[0].fitness)
        if ((i + 1) % PLOT_EVERY_X_GENERATIONS) == 0 or i == 0:
            plot_tsp(population[0].path, 'Best from generation: ' + str(i + 1), population[0].fitness)


def mutation(individual: Individual, mutation_probability: float) -> Individual:
    """
    :param individual: individual that will mutate
    :return: individual that has mutated
    """
    path = individual.path.copy()

    start = 0
    if ALWAYS_START_AT_1:
        start = 1
    for i in range(start, len(path)):
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

    return Individual(get_fitness(path), path)


def convert_to_valid_path(path_child: List[int], cities: List[int]):
    valid_path = []
    forgotten_cities = []

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


def greedy_algorithm(cities: List[int], start: int) -> Individual:
    new_path = [start]
    current_city = start
    for i in range(len(cities) - 1):
        closest_city = []

        for city in cities:

            distance = get_distance_from_instance(current_city, city)

            if not closest_city:
                if city not in new_path:
                    closest_city.append(distance)
                    closest_city.append(city)
            elif closest_city[0] > distance:
                if city not in new_path:
                    closest_city[0] = distance
                    closest_city[1] = city

        new_path.append(closest_city[1])
        current_city = closest_city[1]

    return Individual(get_fitness(new_path), new_path)


def update_population_with_new_children(population, children):
    population.extend(children)


def plus_selection(mu: int, llambda: int, population: List[Individual], mutation_probability, cities, crossovers = 6) -> List[
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

    new_parents = best_parents

    # 2. create lambda children from the mu best parents
    print("New Offspring")
    new_offspring = []
    first_mutation = False
    first_recombination = False
    while len(new_offspring) < llambda:
        random_number = random.random()
        # if < 0.5 -> mutation, if >0.5 -> recombination
        if random_number < 0.5:
            parent = best_parents[random.randint(0, len(best_parents) - 1)]

            if not first_mutation:
                print()
                print("First mutation, mutation probability: " + str(mutation_probability))
                print("Path to mutate: ", parent.path, "Fitness: ", parent.fitness)

            mutated_ind = mutation(parent, mutation_probability)

            if not first_mutation:
                print("Path to mutate: ", mutated_ind.path, "Fitness: ", mutated_ind.fitness)
                first_mutation = True

            new_offspring.append(mutated_ind)
        else:
            parent_1 = best_parents[random.randint(0, len(best_parents) - 1)]
            parent_2 = best_parents[random.randint(0, len(best_parents) - 1)]

            if not first_recombination:
                print()
                print("First Recombination, crossover points: " + str(crossovers))
                print("parent_1 path:", parent_1.path, "Fitness: ", parent_1.fitness)
                print("parent_2 path: ", parent_2.path, "Fitness: ", parent_2.fitness)

            while parent_1.path == parent_2.path:
                parent_1 = best_parents[random.randint(0, len(best_parents) - 1)]

            recombinated_ind = crossover(parent_1,
                                         parent_2,
                                         crossovers,
                                         cities)
            if not first_recombination:
                print("child1 path:", recombinated_ind[0].path, "Fitness: ", recombinated_ind[0].fitness)
                print("child2 path: ", recombinated_ind[1].path, "Fitness: ", recombinated_ind[1].fitness)
                first_recombination = True

            new_offspring.extend(recombinated_ind)

    # 3. choose mu best individuals from parents + children
    mu_lambda_together = new_offspring + best_parents

    if WITH_GREEDY:
        for i in range(10):
            mu_lambda_together.append(greedy_algorithm(cities, random.choice(cities)))

    remove_same_path(mu_lambda_together)
    # sort ascending based on the fitness. 
    mu_best_individuals = sorted(mu_lambda_together, key=lambda individuum: individuum.fitness)
    # choose the best mu - individuals
    mu_best_individuals = mu_best_individuals[:mu]
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


def plot_tsp(path: List[int], caption: str, fitness: int):
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
    title = caption + ", fitness: " + str(fitness)
    plt.title(title)
    plt.show()


def main(n_iteration, best_parents, n_children, mutation_probability):
    # tsp file load
    file = ACTIV_PROBLEM_FILEPATH[0]
    # 2. Testinstanz hinzufügen
    read_tsp_file(file)
    # test if tsp can be read
    cities = get_cities_from_instance()
    distance_between_city1_city2 = get_distance_from_instance(cities[0], cities[1])

    # create population
    population = create_initial_population(cities)
    if ALWAYS_START_AT_1:
        for individual in population:
            individual.path[0] = 1
            individual.path = convert_to_valid_path(individual.path, cities)

    # welche fitness schafft ein einfacher greedy Algoritmus?
    greed = greedy_algorithm(cities, 1)
    print("greedy: " + str(greed.fitness))
    plot_tsp(greed.path, "Greedy algorithm", greed.fitness)

    print("The Evolutionary Process beginns:")
    # Welchen Einfluss haben die Parameter auf die Ergebnisse?
    # mit mehr generationen bekommen wir besseren fitness da wir immer die besten auswählen
    evolutionary_algortihm(population, n_iteration, cities, best_parents, n_children, mutation_probability)
    opt = tsplib95.load(ACTIV_PROBLEM_FILEPATH[1])
    print()
    print('optimal tour: ', opt.tours)
    print('fitness of optimal tour: ', tsp_instance.trace_tours(opt.tours)[0])
    plot_tsp(opt.tours[0], "Optimal tour", tsp_instance.trace_tours(opt.tours)[0])

    plt.plot(BESTFINESS)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    title = "Fitness over Time"
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # immer mit den selben random Zahlen
    random.seed(41)
    # main(10,3,4,0.2)
    # more generations -> better fitness
    # main(10,3,4,0.2)
    # more best parents ->
    ACTIV_PROBLEM_FILEPATH = a280_FILEPATH
    main(100, 120, 600, 0.08)
