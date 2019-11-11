from __future__ import division
import random
import sys
import numpy as np
import math

NR_PARTICLES = 50
MAX_ITERATIONS = 500

# --- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total


# --- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, costFunc, nr_dimensions, fi1, fi2, w, bounds_down, bounds_up):
        self.position = np.random.uniform(low=bounds_down, high=bounds_up, size=nr_dimensions)  # particle position
        self.velocity = np.random.uniform(low=-1, high=1, size=nr_dimensions)  # particle velocity
        self.pos_best = None  # best position individual
        self._group_best = None  # best group position
        self.value_best = sys.float_info.max  # best value individual
        self.value = sys.float_info.max  # value individual

        self._fi1 = fi1  # cognative constant
        self._fi2 = fi2  # social constant
        self._w = w  # inertia weight

        self._costFunc = costFunc
        self._nr_dimensions = nr_dimensions
        self._bounds = {
            'down': bounds_down,
            'up': bounds_up
        }

    # evaluate current function
    def evaluate(self):
        self.value = self._costFunc(self.position)

        # check to see if the current position is an individual best
        if self.value < self.value_best:
            self.pos_best = self.position
            self.value_best = self.value

    @property
    def group_best(self):
        return self._group_best

    @group_best.setter
    def group_best(self, value):
        self._group_best = value

    # update new particle velocity
    def update_velocity(self):

        u1 = np.random.uniform(size=self._nr_dimensions)
        u2 = np.random.uniform(size=self._nr_dimensions)

        self.velocity = self._w * self.velocity + \
                        self._fi1 * u1 * (self.pos_best - self.position) + \
                        self._fi2 * u2 * (self.group_best - self.position)

    # update the particle position based off new velocity updates
    def update_position(self):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, a_min=self._bounds['down'], a_max=self._bounds['up'])

class Topology():
    def __init__(self, type, swarm):
        self._swarm = swarm
        self._type = type

    def update_group_best(self):
        method_to_call = getattr(self, '_update_group_best_'+self._type)
        method_to_call()

    def _update_group_best_full(self):
        # FULL topology update
        group_best_value = sys.float_info.max
        group_best_pos = None

        for particle in self._swarm:
            if particle.value_best <= group_best_value:
                group_best_pos = particle.position
                group_best_value = particle.value_best

        for particle in self._swarm:
            particle.group_best = group_best_pos

    def _update_group_best_ring(self):
        # Ring topology update
        for i in range(0,len(self._swarm)):
            group_best_value = sys.float_info.max
            group_best_pos = None

            if self._swarm[i-1].best_value < group_best_value:
                group_best_value = self._swarm[i-1].value_best
                group_best_pos = self._swarm[i-1].pos_best

            if self._swarm[i].best_value < group_best_value:
                group_best_value = self._swarm[i].value_best
                group_best_pos = self._swarm[i].pos_best

            if self._swarm[(i+1)%len(self._swarm)].best_value < group_best_value:
                group_best_value = self._swarm[(i+1)%len(self._swarm)].value_best
                group_best_pos = self._swarm[(i+1)%len(self._swarm)].pos_best

            self._swarm[i].group_best = group_best_pos

    def _update_group_best_4neighbours(self):
        # 4 neighbours topology update
        for i in range(0,len(self._swarm)):
            group_best_value = sys.float_info.max
            group_best_pos = None

            if self._swarm[i-2].best_value < group_best_value:
                group_best_value = self._swarm[i-2].value_best
                group_best_pos = self._swarm[i-2].pos_best

            if self._swarm[i-1].best_value < group_best_value:
                group_best_value = self._swarm[i-1].value_best
                group_best_pos = self._swarm[i-1].pos_best

            if self._swarm[i].best_value < group_best_value:
                group_best_value = self._swarm[i].value_best
                group_best_pos = self._swarm[i].pos_best

            if self._swarm[(i+1)%len(self._swarm)].best_value < group_best_value:
                group_best_value = self._swarm[(i+1)%len(self._swarm)].value_best
                group_best_pos = self._swarm[(i+1)%len(self._swarm)].pos_best

            if self._swarm[(i+2)%len(self._swarm)].best_value < group_best_value:
                group_best_value = self._swarm[(i+2)%len(self._swarm)].value_best
                group_best_pos = self._swarm[(i+2)%len(self._swarm)].pos_best

            self._swarm[i].group_best = group_best_pos



class PSO():
    def __init__(self, topology_type, costFunc, nr_dimensions,  fi1, fi2, w, bounds_down, bounds_up):
        # establish the swarm
        self.swarm = []
        for i in range(0, NR_PARTICLES):
            self.swarm.append(Particle(costFunc, nr_dimensions,  fi1, fi2, w, bounds_down, bounds_up))

        self.topology = Topology(topology_type, self.swarm)

    def run(self):
        # begin optimization loop
        i = 0
        while i < MAX_ITERATIONS:

            for particle in self.swarm:
                particle.evaluate()

            self.topology.update_group_best()

            for particle in self.swarm:
                particle.update_velocity()
                particle.update_position()

            i += 1

        best_value = sys.float_info.max
        best_position = None
        for particle in self.swarm:
            if particle.value_best < best_value:
                best_value = particle.value_best
                best_position = particle.pos_best

        return best_position, best_value

# --- RUN ----------------------------------------------------------------------+

best_pos, best_val = PSO(
    topology_type='ring', costFunc=func1, nr_dimensions=2,
    fi1=1, fi2=2, w=0.4, bounds_down=[-10,-10], bounds_up=[10,10]
).run()

print(best_pos)
print(best_val)