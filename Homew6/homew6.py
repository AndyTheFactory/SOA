
import random
import sys
import numpy as np
import math

# --- COST FUNCTIONS ------------------------------------------------------------+

# functions we are attempting to optimize (minimize)
def func_spehre(x):
    #bounds [-100,100]
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

def func_rosenbrock(x):
    #bounds [-30,30]
    total = 0
    for i in range(len(x)-1):
        total += 100 * ((x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
    return total

def func_rastrigin(x):
    #bounds [-5.12,5.12]
    total = 0
    for i in range(len(x)):
        total += (x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return total

def func_griewank(x):
    #bounds [-600,600]
    total = 0
    mult = 1
    for i in range(len(x)):
        total += x[i] ** 2
        mult *= math.cos( x[i] / (i+1))
    return 1 + total / 4000 - mult

NR_CATS = 20
SMP = 10        # seeking memory pool
SRD = 0.7       # seeking range of the selected dimension
CDC = 2         # counts of dimensions to change


class Cat:
    def __init__(self, postition, velocity, costFunc, mode):
        self.postition = postition
        self.velocity = velocity
        self.costFunc = costFunc
        self.mode = mode





class Dispatcher:
    def __init__(self,):
        pass
    