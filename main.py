import random
from typing import List
import tsplib95

# define own typefor the test instance from tsplib
TSP_INSTANCE =  tsplib95.models.Problem

# population size
POPULATION_SIZE = 10

def read_tsp_file(file: str) -> TSP_INSTANCE:
    """
    read tsp test instance from tsplib
    return: problem class which has all nodes(=cities) and weights(=distances)
    """
    return tsplib95.load(file)

class Individual():
    # initialize
    def __init__(self, fitness: int, path: list[int]) -> None:
        # fitness = distance between cities
        self.fitness = fitness
        # path = list of the cities which have been/can be visited
        self.path = path

def get_cities_from_instance(tsp_instance: TSP_INSTANCE):
    return list(tsp_instance.get_nodes())

def get_distance_from_instance(tsp_instance: TSP_INSTANCE, city1: int, city2: int):
    return tsp_instance.get_weight(city1,city2)

def get_fitness(path: list[int], tsp_instance: TSP_INSTANCE):
    """

    :param path: the list of citites to visit
    :param tsp_instance: tsp test instance
    :return: sum of the distances = fitness
    """
    fitness = 0
    for i in range(len(path)-1):
        fitness += get_distance_from_instance(tsp_instance,path[i],path[i+1])
    return fitness

def get_random_path(cities):
    """
    choose a random order of cities
    :return: list of cities
    """
    random.shuffle(cities)
    return cities

def create_initial_population(cities, tsp_instance)->List[Individual]:
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
        fitness = get_fitness(path, tsp_instance)
        population.append(Individual(fitness, path))


    # print("Population: ")
    # for i in range(POPULATION_SIZE):
    #     print(population[i].path, population[i].fitness);
    return population

def evolutionary_algortihm(population: List[Individual], n_iterations: int):
    """
    while loop for many generations
    """
    pass

def mutation(individual : Individual, mutation_probability: float) -> Individual:
    """

    :param individual: individual that will mutate
    :return: individual that has mutated
    """
    path = individual.path
    print("Path to mutate: ", path,"Fitness: ", individual.fitness)
    temp = 0
    # number between 0 and 1
    for i in range(len(path)):
        random_number = random.random()
        if (random_number < mutation_probability):
            # the city with a random number < mutation_probability is being replaced
            # with a random city from the same path so that
            # it is ensured that there is no double occurance of the same city
            temp = path[i]
            new_city = random.choice(path)
            index_new_city = path.index(new_city)
            path[i] = new_city
            path[index_new_city] = temp
    # save the new path in the individual
    individual.path = path
    # ändert sich der Fitness von alleine?
    print("Mutated path: ", path,"Fitness: ",  individual.fitness)
    return individual


def convert_to_valid_path(path_child: List[int], cities: List[int]):
    valid_path = []
    # all cities from the build child path but only once
    for city in path_child:
        if city not in valid_path:
            valid_path.append(city)
    # all cities that are missing in the path
    for city in cities:
        if city not in valid_path:
            valid_path.append(city)

    return valid_path


def recombination(individual1: Individual, individual2: Individual, cross_point: int, tsp_instance, cities) -> List[Individual]:
    """

    :param individual1: parent one
    :param individual2:  parent two
    :return: recombinated child
    """

    print("I1 path:", individual1.path)
    print("I2 path: ", individual2.path)
    #subset the paths in two with the crossover point
    individual1_subpath_first_half = individual1.path[:cross_point]
    individual1_subpath_second_half = individual1.path[cross_point:]
    individual2_subpath_first_half = individual2.path[:cross_point]
    individual2_subpath_second_half = individual2.path[cross_point:]

    #build the children paths
    path_child1 = individual1_subpath_first_half + individual2_subpath_second_half
    path_child2 = individual2_subpath_first_half + individual1_subpath_second_half
    # recombined paths should only contain each city once
    path_child1 = convert_to_valid_path(path_child1,cities)
    path_child2 = convert_to_valid_path(path_child2, cities)

    fitness_child1  = get_fitness(path_child1,tsp_instance )
    fitness_child2 = get_fitness(path_child2, tsp_instance)
    #print the offspring :D
    print("child1 path:",  path_child1, "Fitness: ", fitness_child1)
    print("child2 path: ", path_child2,"Fitness: ", fitness_child2)

    child1 = Individual(fitness_child1, path_child1)
    child2 = Individual(fitness_child2, path_child2)

    return [child1, child2]


def update_population_with_new_children(population, children):
    population.extend(children)

def plus_selection(mu: int, llambda: int, population: List[Individual], mutation_probability) -> List[Individual]:
    """
    1. choose best mu parents
    2. create lambda children from the mu best parents
    3. choose mu best individuals from parents + children

    :param mu: number of individuals for the new population
    :param llambda: number of children to create of the old population
    :param population: old population
    :return: new population with mu indivduals
    """
    # if <0.5 -> mutation, if >0.5 -> recombination

    # 1. choose best mu parents
    best_parents = []
    for i in range(len(population)-1):
        if(population[i].fitness>population[i+1].fitness):
            best_parents.append(population[i])
    for parent in best_parents:
        print(parent.path, parent.fitness)

def print_population(population):
    print("\nPopulation: \nPATH	 FITNESS VALUE\n")
    for i in range(len(population)):
        print(population[i].path, population[i].fitness)
    print()


def main():
    # tsp file load
    file = "C:/Daten/Studium Info/Informatik TH Köln/5.Sem/KI/P/tsp-Dateien/a280.tsp"
    tsp_test_instance = read_tsp_file(file)
    #test if tsp can be read
    cities = get_cities_from_instance(tsp_test_instance)
    distance_between_city1_city2 = get_distance_from_instance(tsp_test_instance,cities[0], cities[1])
    print("Cities: ",cities)
    print("Distance between first and second city: ", distance_between_city1_city2)

    # create population
    population = create_initial_population(cities,tsp_test_instance)
    print_population(population)

    # mutation
    mutation(population[0], 0.2)

    # recombination
    offspring = recombination(population[1],population[2], 3, tsp_test_instance,cities)
    # update the population with the new children
    update_population_with_new_children(population,offspring)
    print("Population after recombination")
    # still not 12
    print_population(population)

    print("Selektion")
    plus_selection(0,0,population,0.2)

if __name__ == "__main__":
    main()
