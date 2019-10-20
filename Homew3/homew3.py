import numpy as np
import random
import sys

global G1, G2, G3


def init_array(size):
    A = np.random.randint(100, size=(9, 9))
    np.fill_diagonal(A, 0)
    A = np.triu(A) + np.tril(A.T)
    return A


def TSP(Graph, S, destination):
    if (len(S) == 2):
        lS = list(S)
        return list([(1, destination)]), Graph[lS[0], lS[1]]
    else:
        min_dist = 999
        min_node = -1
        min_way = list()
        for _, node in enumerate(S):
            if (node != 1 and node != destination):
                w, d = TSP(Graph, S.difference({node}), node)
                d += Graph[node - 1, destination - 1]
                if d < min_dist:
                    min_dist = d
                    min_node = node
                    min_way = w
                    min_way.append((node, destination))
        return min_way, min_dist


class Ant:
    def __init__(self, colony):
        self.colony = colony
        self.tabu = list()
        self.all_nodes = set()
        self.position = 0
        self.L = 0

    def probability(self, j):
        if j in self.tabu:
            return 0
        denominator = 0.0
        for k in self.all_nodes - set(self.tabu):
            denominator += self.colony.tau[self.position, k] ** self.colony.alpha * \
                           (1 / self.colony.Graph[self.position, k]) ** self.colony.beta
        p: float
        p = self.colony.tau[self.position, j] ** self.colony.alpha * \
            (1 / self.colony.Graph[self.position, j]) ** self.colony.beta
        return p

    def move(self):
        prob = [self.probability(x) for x in self.all_nodes]
        nextpos = np.random.choice(self.all_nodes, p=prob)
        self.tabu.append(nextpos)
        self.L += self.colony.Graph[self.position, nextpos]
        self.position=nextpos
        return nextpos

    def close_circuit(self):
        if self.all_nodes - set(self.tabu) == set():
            self.L += self.colony.Graph[self.position, self.tabu[0]]

    def get_delta_tau(self):
        delta_tau_k = np.zeros(self.colony.Graph.shape)
        for i,node in enumerate(self.tabu):
            next_node = self.tabu[i+1] if i < len(self.tabu)-1 else self.tabu[0]
            delta_tau_k[node, next_node] = AntColony.Q / self.L
            delta_tau_k[next_node, node] = AntColony.Q / self.L


class AntColony:
    q = 1.5
    Q = 0.1
    nr_ants = 5
    N_MAX = 1000
    TAU_ZERO = 1e-5

    def __init__(self, alpha, beta, rho):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.ants = list()
        self.tau = None
        self.nr_nodes = None
        for i in range(self.nr_ants):
            self.ants.append(Ant(self))

    def loadGraph(self, Graph):
        self.Graph = Graph
        self.tau = np.zeros(Graph.shape) + self.TAU_ZERO
        self.delta_tau = np.zeros(Graph.shape)
        self.nr_nodes = Graph.shape[0]
        for ant in self.ants:
            while True:
                pos = random.randint(0, self.nr_nodes-1)  # Nodes are 1..N , Graph is 0..N-1
                if not self.checkPosition(pos):
                    break
            ant.position = pos
            ant.tabu.append(pos)
            ant.all_nodes = set(range(1, self.nr_nodes + 1))

    def checkPosition(self, node):
        for ant in self.ants:
            if ant.position == node:
                return True
        return False

    def getBestCircuit(self):
        min = sys.maxsize
        min_way = list()
        for ant in self.ants:
            if ant.L < min:
                min_way=ant.tabu
                min=ant.L

        return min_way, min

    def checkStagnation(self):
        prev=None
        for ant in self.ants:
            if prev==None:
                continue
            if prev.L != ant.L or not np.array_equal(np.array(prev.tabu),np.array(ant.tabu)):
                return False
        return True

    def move_all_ants(self):
        for n in range(0, self.nr_nodes):
            for ant in self.ants:
                ant.move()
        for ant in self.ants:
            ant.close_circuit()

    def update_delta_tau(self):
        for ant in self.ants:
            self.delta_tau += ant.get_delta_tau()

    def run(self):
        n_iteration = 0
        best_way = list()
        best_cost = sys.maxsize
        while True:
            self.move_all_ants()
            way, cost = self.getBestCircuit()
            if cost<best_cost:
                best_cost = cost
                best_way = way

            self.update_delta_tau()

            self.tau = self.tau * self.rho + self.delta_tau
            n_iteration += 1

            self.delta_tau = np.zeros(self.Graph.shape)

            if n_iteration >= self.N_MAX or self.checkStagnation():
                break

        return best_way, best_cost


if __name__ == "__main__":
    G1 = init_array(9)
    print(TSP(G1,set(range(1,9)),9))
    ac = AntColony(1, 1, 1)
    ac.loadGraph(G1)
