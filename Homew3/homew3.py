import numpy as np
import random
import sys
import tqdm

global G1, G2, G3


def init_array(size):
    A = np.random.randint(low=1, high=100, size=(size, size))
    np.fill_diagonal(A, 0)
    A = np.triu(A) + np.tril(A.T)
    return A


def TSP_ph1(Graph, S, destination):
    if (len(S) <= 0):
        # s=list(S)
        return list([(1, destination)]), Graph[0, destination - 1]
    else:
        min_dist = sys.maxsize
        min_way = list()
        S = S.difference({destination})
        for _, node in enumerate(S):
            w, d = TSP_ph1(Graph, S.difference({node}), node)
            d += Graph[node - 1, destination - 1]
            if d < min_dist:
                min_dist = d
                min_way = w
                min_way.append((node, destination))
        return min_way, min_dist


def TSP(Graph):
    nodecount = len(Graph[0])
    S = set(range(2, nodecount + 1))
    min_dist = sys.maxsize
    min_way = list()
    for _, node in enumerate(S):
        w, d = TSP_ph1(Graph, S, node)
        d += Graph[node - 1, 0]
        if d < min_dist:
            min_dist = d
            min_way = w
            min_way.append((node, 1))
    return min_way, min_dist


class Ant:
    def __init__(self, colony):
        self.colony = colony
        self.tabu = list()
        self.all_nodes = set()
        self.position = -1
        self.L = 0

    def probability(self, j):
        if j in self.tabu:
            return 0
        denominator = sys.float_info.epsilon
        for k in self.all_nodes.difference(set(self.tabu)):
            denominator += self.colony.tau[self.position, k] ** self.colony.alpha * \
                           (1 / self.colony.Graph[self.position, k]) ** self.colony.beta
        p: float
        p = self.colony.tau[self.position, j] ** self.colony.alpha * \
            (1 / self.colony.Graph[self.position, j]) ** self.colony.beta
        return p / denominator

    def move(self):
        prob = [self.probability(x) for x in self.all_nodes]
        s = sum(prob)
        if s != 1.:
            prob = [x / s for x in prob]
        nextpos = np.random.choice(list(self.all_nodes), p=prob)
        self.tabu.append(nextpos)
        self.L += self.colony.Graph[self.position, nextpos]
        self.position = nextpos
        return nextpos

    def close_circuit(self):
        if self.all_nodes.difference(set(self.tabu)) == set():
            self.L += self.colony.Graph[self.position, self.tabu[0]]

    def get_delta_tau(self):
        delta_tau_k = np.zeros(self.colony.Graph.shape).astype(float)
        for i, node in enumerate(self.tabu):
            next_node = self.tabu[i + 1] if i < len(self.tabu) - 1 else self.tabu[0]
            delta_tau_k[node, next_node] = AntColony.Q / self.L
            delta_tau_k[next_node, node] = AntColony.Q / self.L
        return delta_tau_k


class AntColony:
    q = 1.5
    Q = 100
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
                pos = random.randint(0, self.nr_nodes - 1)  # Nodes are 1..N , Graph is 0..N-1
                if not self.checkPosition(pos):
                    break
            ant.position = pos
            ant.tabu.append(pos)
            ant.all_nodes = set(range(0, self.nr_nodes))

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
                min_way = ant.tabu
                min = ant.L

        return min_way, min

    def checkStagnation(self):
        prev = None
        for ant in self.ants:
            if prev == None:
                prev = ant
                continue
            if prev.L != ant.L or not np.array_equal(np.array(prev.tabu), np.array(ant.tabu)):
                return False
            prev = ant
        return True

    def move_all_ants(self):
        for n in range(0, self.nr_nodes - 1):  # -1 because we initalize the ants with one node
            for ant in self.ants:
                ant.move()
        for ant in self.ants:
            ant.close_circuit()

    def update_delta_tau(self):
        for ant in self.ants:
            self.delta_tau += ant.get_delta_tau()

    def reset_ants(self):
        for ant in self.ants:
            ant.position = ant.tabu[0]
            ant.tabu = [ant.position]
            ant.L = 0

    def run(self):
        n_iteration = 0
        best_way = list()
        best_cost = sys.maxsize
        pbar = tqdm.tqdm(total=self.N_MAX)

        while True:
            self.move_all_ants()
            way, cost = self.getBestCircuit()
            if cost < best_cost:
                best_cost = cost
                best_way = way

            self.update_delta_tau()

            self.tau = self.tau * self.rho + self.delta_tau
            n_iteration += 1
            pbar.update(1)

            self.delta_tau = np.zeros(self.Graph.shape).astype(float)

            if n_iteration >= self.N_MAX or self.checkStagnation():
                break
            self.reset_ants()

        pbar.close()
        res_way = []
        p = None
        for n in best_way:
            if p == None:
                p = n
                continue
            res_way.append((p + 1, n + 1))
            p = n
        res_way.append((best_way[-1] + 1, best_way[0] + 1))
        return res_way, best_cost


if __name__ == "__main__":
    G1 = init_array(9)
    # G1 = np.array([[0, 43, 18, 34, 67, 41, 32, 1, 84],
    #                [43, 0, 93, 51, 1, 86, 44, 92, 27],
    #                [18, 93, 0, 49, 33, 87, 36, 22, 2],
    #                [34, 51, 49, 0, 18, 9, 24, 19, 28],
    #                [67, 1, 33, 18, 0, 89, 85, 64, 57],
    #                [41, 86, 87, 9, 89, 0, 27, 9, 5],
    #                [32, 44, 36, 24, 85, 27, 0, 42, 10],
    #                [1, 92, 22, 19, 64, 9, 42, 0, 44],
    #                [84, 27, 2, 28, 57, 5, 10, 44, 0]])

    print(TSP(G1))
    G1 = np.array([[ 0, 92, 30,  0, 25, 83, 26, 62, 36],
       [92,  0, 44, 76, 65, 81, 95, 24, 28],
       [30, 44,  0, 85, 54, 74, 49, 37, 70],
       [ 0, 76, 85,  0, 25,  0, 63, 40,  8],
       [25, 65, 54, 25,  0, 90, 77, 79, 94],
       [83, 81, 74,  0, 90,  0, 87, 57, 48],
       [26, 95, 49, 63, 77, 87,  0, 13, 69],
       [62, 24, 37, 40, 79, 57, 13,  0, 48],
       [36, 28, 70,  8, 94, 48, 69, 48,  0]])

    ac = AntColony(0.5, 1, 0.3)
    ac.loadGraph(G1)
    print("AC Result: ")
    way, cost = ac.run()
    print(way, cost)
