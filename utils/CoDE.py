import numpy as np
from tqdm import tqdm

from utils.DE import DE_individual, best_fitness, rand_1_bin, selection, init_pop

def rand_2_bin(pop, iteration, mutationRate, scaling_factor, problem_size, _min, _max):
    # V_i, g = X_{r1, g} + F * (X_{r2, g} - X_{r3, g}) + F * (X_{r4, g} - X_{r5, g})

    available_choices = [idx for idx in range(len(pop)) if idx != iteration]

    r1, r2, r3, r4, r5 = np.random.choice(available_choices, size=5, replace=False)
    j_rand = np.random.choice(problem_size)

    mask = np.random.rand(problem_size) < mutationRate
    mask[j_rand] = True

    recomb = np.clip((pop[r1].x +
                     scaling_factor * (pop[r2].x - pop[r3].x) +
                     scaling_factor * (pop[r4].x - pop[r5].x)), _min, _max)

    return np.where(mask, recomb, pop[iteration].x)

def current_to_rand_1(pop, iteration, mutationRate, scaling_factor, problem_size, _min, _max):

    available_choices = [idx for idx in range(len(pop)) if idx != iteration]
    r1, r2, r3 = np.random.choice(available_choices, size=3, replace=False)
    mask = np.random.rand(problem_size)

    # trl[j] = trgt[j] + k * (dnr1[j] - trgt[j]) + sf * (dnr2[j] - dnr3[j])

    trial = np.clip(pop[iteration].x + \
                    mask * (pop[r1].x - pop[iteration].x) + \
                    scaling_factor * (pop[r2].x - pop[r3].x), _min, _max)

    return trial

def CoDE(pair,
         init_population,
         problem_size,
         scaling_factor, #list of three numbers
         crossover_rate, #list of three numbers
         lower_bound,
         upper_bound,
         generation_max,
         scores=None):
    
    obj_fn = pair.obj_fun
    
    if scores is None:
        scores = {}
    pop_size = pair.popsize
    
    scaling_factor = [1.0, 1.0, 0.8]
    crossover_rate = [0.1, 0.9, 0.2]
    
    np.clip(init_population, lower_bound, upper_bound, out=init_population)

    population = [DE_individual(p) for p in init_population]
    candidates = [DE_individual(np.zeros(problem_size)) for _ in range(pop_size)]

    solution = 0

    for individual in tqdm(population):
        individual.fitness = obj_fn(individual.x, problem_size)

    for generation in range(generation_max):
        for j in tqdm(range(pop_size)):
            trial1 = rand_1_bin(population, j,
                                   crossover_rate[0],
                                   scaling_factor[0],
                                   problem_size,
                                   lower_bound, upper_bound)
            trial2 = rand_2_bin(population, j,
                                   crossover_rate[1],
                                   scaling_factor[1],
                                   problem_size,
                                   lower_bound, upper_bound)
            trial3 = current_to_rand_1(population, j,
                                       crossover_rate[2],
                                       scaling_factor[2],
                                       problem_size,
                                       lower_bound, upper_bound)

            _1 = DE_individual(trial1)
            _1.fitness = obj_fn(trial1, problem_size)
            _2 = DE_individual(trial2)
            _2.fitness = obj_fn(trial2, problem_size)
            _3 = DE_individual(trial3)
            _3.fitness = obj_fn(trial3, problem_size)

            trial_vecs = [_1, _2, _3]

            best_trial = sorted(trial_vecs, key=lambda x: x.fitness)[0]

            candidates[j].x = best_trial.x
            candidates[j].fitness = best_trial.fitness


        for k, c in enumerate(candidates):
            selection(population[k], c)

        scores[generation] = [p.fitness for p in population]

    solution += best_fitness(population)
    # print(solution)
    return sorted(population, key=lambda x: x.fitness)[0].x

def sphere(x, dim):
    return np.dot(x, x)

if __name__ == "__main__":
    ans = CoDE(obj_fn=sphere,
               pop_size=100,
               problem_size=30,
               scaling_factor=0.5,
               crossover_rate=0.5,
               lower_bound=-100,
               upper_bound=100,
               generation_max=1000)

    print(sphere(ans))
