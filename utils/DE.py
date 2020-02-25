import sys
import numpy as np

class DE_individual:
    def __init__(self, x):
        self.x = x
        self.fitness = np.inf

def init_pop(population, _min, _max):
    dim = len(population[0].x)
    for p in population:
        p.x = (_max - _min) * np.random.uniform(size=dim) + _min
        # LOWER_BOUND + (rand() / (double)RAND_MAX) * (UPPER_BOUND - LOWER_BOUND)
    return

def rand_1_bin(pop, cand, iteration, mutationRate, scaling_factor, problem_size, _min, _max):

    available_choices = [idx for idx in range(len(pop)) if idx != iteration]

    r1, r2, r3 = np.random.choice(available_choices, size=3, replace=False)
    j_rand = np.random.choice(problem_size)

    mask = np.random.rand(problem_size) < mutationRate
    mask[j_rand] = True

    recomb =  np.clip(pop[r1].x + scaling_factor * (pop[r2].x - pop[r3].x), _min, _max)

    cand[iteration].x = np.where(mask, recomb, pop[iteration].x)


def selection(pop_member, candidate_member):
    if candidate_member.fitness <= pop_member.fitness:
        pop_member.x = candidate_member.x
        pop_member.fitness = candidate_member.fitness

    return

def best_fitness(population):
    return sorted(population, key=lambda x: x.fitness)[0].fitness

def DE(obj_fn,
         pop_size,
         problem_size,
         scaling_factor,
         crossover_rate,
         lower_bound,
         upper_bound,
         generation_max):

    population = [DE_individual(np.zeros(problem_size)) for _ in range(pop_size)]
    candidates = [DE_individual(np.zeros(problem_size)) for _ in range(pop_size)]

    solution = 0

    init_pop(population, lower_bound, upper_bound)
    for individual in population:
        individual.fitness = obj_fn(individual.x)

    for generation in range(generation_max):
        for j in range(pop_size):
            rand_1_bin(population, candidates, j,
                       crossover_rate,
                       scaling_factor,
                       problem_size,
                       lower_bound, upper_bound)


        for k, c in enumerate(candidates):
            c.fitness = obj_fn(c.x)
            selection(population[k], c)

    solution += best_fitness(population)
    print(solution)
    return sorted(population, key=lambda x: x.fitness)[0].x

def sphere(x):
    return np.dot(x, x)

if __name__ == "__main__":
    ans = DE(obj_fn=sphere,
               pop_size=100,
               problem_size=30,
               scaling_factor=0.5,
               crossover_rate=0.5,
               lower_bound=-100,
               upper_bound=100,
               generation_max=1000)
    print(sphere(ans))