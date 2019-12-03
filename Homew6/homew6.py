import random
import sys
from copy import deepcopy

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --- COST FUNCTIONS ------------------------------------------------------------+

# functions we are attempting to optimize (minimize)
def func_spehre(x):
    # bounds [-100,100]
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total


def func_rosenbrock(x):
    # bounds [-30,30]
    total = 0
    for i in range(len(x) - 1):
        total += 100 * ((x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
    return total


def func_rastrigin(x):
    # bounds [-5.12,5.12]
    total = 0
    for i in range(len(x)):
        total += (x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return total


def func_griewank(x):
    # bounds [-600,600]
    total = 0
    mult = 1
    for i in range(len(x)):
        total += x[i] ** 2
        mult *= math.cos(x[i] / (i + 1))
    return 1 + total / 4000 - mult


NR_CATS = 20
SMP = 20  # seeking memory pool
SRD = 0.15  # seeking range of the selected dimension
CDC = 2  # counts of dimensions to change
SPC = True  # Self-Position Consideration
C1 = 0.4  # Tracing mode constant
MR = 0.2  # mixture ratio

V_MAX = 100
MAX_ITERATIONS = 50

IS_MINIMIZATION_PROBLEM = True


class Cat:
    def __init__(self, position, velocity, costFunc, mode):
        self.position = position
        self.velocity = velocity
        self.costFunc = costFunc
        self.mode = mode

        self.best_cat_pos = position

    def act(self):
        if self.mode == 'seeking':
            self.seeking()

        if self.mode == 'tracing':
            self.tracing()

    def seeking(self):

        scouts = list([])
        fitness = np.zeros(SMP)

        for j in range(SMP):
            if j == SMP - 1 and SPC:
                scouts.append(deepcopy(self))
            else:
                scouts.append(
                    Cat(self._random_position(), self.velocity, self.costFunc, self.mode)
                )
            # scouts[-1].position += scouts[-1].velocity
            fitness[j] = scouts[-1].fitness()


        FS_max = np.max(fitness)
        FS_min = np.min(fitness)

        if IS_MINIMIZATION_PROBLEM:
            FS_b = FS_max
        else:
            FS_b = FS_min

        probabilities = np.ones(SMP)

        if FS_min != FS_max:
            probabilities = np.array([
                (abs(FS_i - FS_b) / (FS_max - FS_min)) for FS_i in fitness
            ])

        probabilities = probabilities / np.sum(probabilities)

        cat_picked = np.random.choice(range(SMP), p=probabilities)

        self.position = scouts[cat_picked].position  # set new position
        self.velocity = scouts[cat_picked].velocity  # set new position

    def tracing(self):
        #V_MAX = (1 + SRD) * self.velocity
        #V_MIN = (1 - SRD) * self.velocity

        r1 = random.random()
        self.velocity = self.velocity + r1 * C1 * (self.best_cat_pos - self.position)

        self.velocity = np.clip(self.velocity, a_min=(-1 * V_MAX), a_max=V_MAX)

        self.position = self.position + self.velocity

    def fitness(self):
        return self.costFunc(self.position)

    def _random_velocity(self):
        v_new = deepcopy(self.velocity)
        dim_changed = random.sample(range(0, self.position.shape[0]), CDC)
        for d in dim_changed:
            v_new[d] = v_new[d] * (1 + SRD * random.uniform(-1, 1))  # v_new  between [v - SRD * v, v + SRD * v]

        return v_new

    def _random_position(self):
        pos_new = deepcopy(self.position)
        dim_changed = random.sample(range(0, self.position.shape[0]), CDC)
        for d in dim_changed:
            pos_new[d] = pos_new[d] * (1 + SRD * random.uniform(-1, 1))  # v_new  between [v - SRD * v, v + SRD * v]

        return pos_new

class Dispatcher:
    def __init__(self, costFunc, nr_dimensions, bounds_down, bounds_up):

        self.costFunc = costFunc

        self.cats = list([])
        for i in range(NR_CATS):
            positions = np.random.uniform(low=bounds_down, high=bounds_up, size=nr_dimensions)
            velocities = np.random.uniform(low=-1, high=1, size=nr_dimensions) * SRD * positions
            cat = Cat(
                position=positions,
                velocity=velocities,
                costFunc=costFunc,
                mode='tracing' if i <= NR_CATS * MR else 'seeking'
            )
            self.cats.append(cat)

    def update_best_cat(self):
        best_value = sys.float_info.max if IS_MINIMIZATION_PROBLEM else -sys.float_info.max
        best_cat_pos = None
        for cat in self.cats:
            current_val = cat.fitness()
            if IS_MINIMIZATION_PROBLEM:
                if current_val < best_value:
                    best_cat_pos = deepcopy(cat.position)
                    best_value = current_val
            else:
                if current_val > best_value:
                    best_cat_pos = deepcopy(cat.position)
                    best_value = current_val

        for cat in self.cats:
            cat.best_cat_pos = best_cat_pos

        return best_cat_pos

    def run(self, nr_iterations):
        history = {"positions":[], "step":[], "value":[]}
        for i in range(nr_iterations):

            bc = self.update_best_cat()


            for cat in self.cats:
                cat.act()
                history["positions"].append(cat.position)
                history["step"].append(i)
                history["value"].append(self.costFunc(cat.position))

            random.shuffle(self.cats)

            for i in range(NR_CATS):
                self.cats[i].mode = 'tracing' if i <= NR_CATS * MR else 'seeking'

        best_pos = self.update_best_cat()

        return best_pos, self.costFunc(best_pos), history

def plot_history(hist):
    colors = cm.rainbow(np.linspace(0, 1, 1 + max(hist['step'])))

    c = [colors[x] for x in hist["step"]]
    x = [x[0] for x in hist["positions"]]
    y = [x[1] for x in hist["positions"]]

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.scatter(x,y , c=c)
    plt.show()

if __name__ == "__main__":
    best_pos, best_value, hist = Dispatcher(costFunc=func_spehre, nr_dimensions=2, bounds_down=-100, bounds_up=100).run(nr_iterations=50)
    print("Point  = (%s)" % best_pos)
    print("Value = %.5f" % best_value)

    # plot_history(hist)
    # print(Dispatcher(costFunc=func_spehre, nr_dimensions=2, bounds_down=-100, bounds_up=100).run(nr_iterations=100))
    # print(Dispatcher(costFunc=func_spehre, nr_dimensions=2, bounds_down=-100, bounds_up=100).run(nr_iterations=500))

# def func_spehre(x):
#     # bounds [-100,100]
# def func_rosenbrock(x):
#     # bounds [-30,30]
# def func_rastrigin(x):
#     # bounds [-5.12,5.12]
# def func_griewank(x):
#     # bounds [-600,600]
