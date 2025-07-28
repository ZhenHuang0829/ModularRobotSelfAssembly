import time

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import random

class ConnectionGraph:
    def __init__(self, n):
        self.num_nodes = n
        self.graph = nx.Graph()
        h = 925
        r = h / math.sin(math.radians(60))
        self.final_center_point = np.array(
            [[0, 0], [0, 2 * h], [1.5 * r, h], [1.5 * r, -h], [0, -2 * h], [-1.5 * r, -h], [-1.5 * r, h],
             [-1.5 * r, 3 * h], [0, 4 * h], [1.5 * r, 3 * h], [3 * r, 2 * h], [3 * r, 0], [3 * r, -2 * h],
             [1.5 * r, -3 * h], [0, -4 * h], [-1.5 * r, -3 * h], [-3 * r, -2 * h], [-3 * r, 0], [-3 * r, 2 * h],
             [-3 * r, 4 * h], [-1.5 * r, 5 * h], [0, 6 * h], [1.5 * r, 5 * h], [3 * r, 4 * h], [4.5 * r, 3 * h],
             [4.5 * r, h], [4.5 * r, -h], [4.5 * r, -3 * h], [3 * r, -4 * h], [1.5 * r, -5 * h], [0, -6 * h],
             [-1.5 * r, -5 * h], [-3 * r, -4 * h], [-4.5 * r, -3 * h], [-4.5 * r, -h], [-4.5 * r, h],
             [-4.5 * r, 3 * h]])

    def configFullConnection(self):
        h = 925

        # compute distant table and adjacent matrix
        dist_table = np.zeros((len(self.final_center_point), len(self.final_center_point)))
        adjacent_m = np.zeros((len(self.final_center_point), len(self.final_center_point)))
        epsilon = 10
        for i in range(len(self.final_center_point)):
            for j in range(len(self.final_center_point)):
                delta_x = self.final_center_point[j][0] - self.final_center_point[i][0]
                delta_y = self.final_center_point[j][1] - self.final_center_point[i][1]
                dist_table[i][j] = math.sqrt(delta_x ** 2 + delta_y ** 2)
                if i != j and dist_table[i][j] <= 2 * h + epsilon:
                    if delta_x > epsilon:
                        if delta_y > epsilon:
                            adjacent_m[i][j] = 2
                        elif delta_y < -epsilon:
                            adjacent_m[i][j] = 3
                    elif delta_x < -epsilon:
                        if delta_y > epsilon:
                            adjacent_m[i][j] = 6
                        elif delta_y < -epsilon:
                            adjacent_m[i][j] = 5
                    else:
                        if delta_y > epsilon:
                            adjacent_m[i][j] = 1
                        elif delta_y < -epsilon:
                            adjacent_m[i][j] = 4

        for i in range(self.num_nodes):
            self.graph.add_node(i, pos=(self.final_center_point[i][0], self.final_center_point[i][1]))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adjacent_m[i][j] > 0.0:
                    self.graph.add_edge(i, j, relation=adjacent_m[i][j])

    def plot(self):
        nx.draw(self.graph, self.final_center_point, with_labels=True)
        plt.show()

class Astar:
    def __init__(self, initial_num, cg: ConnectionGraph, total_num):
        self.initial_num = initial_num
        self.total_num = total_num
        self.graph = nx.Graph()
        self.cg = cg
        for i in range(self.initial_num):
            self.graph.add_node(i, pos=(cg.final_center_point[i][0], cg.final_center_point[i][1]))

        self.assembled = [0]
        self.assigned = [0]
        self.available = []
        self.paths = []

    def plot(self):
        nx.draw(self.graph, self.cg.final_center_point, with_labels=True)
        plt.show()

    def find_min_length_element(self, lst):
        # 找到最小长度
        min_length = min(len(item) for item in lst)

        # 找到所有最小长度的元素
        min_length_elements = [item for item in lst if len(item) == min_length]

        # 随机选择一个最小长度的元素
        return random.choice(min_length_elements)

    def run(self):
        cg = self.cg

        s = 0
        while True:
            if s == self.total_num-1:
                break

            # Compute assembled nodes
            if len(self.paths) != 0:
                for p in self.paths:
                    if len(p) == s+1:
                        self.assembled.append(p[-1])

            # Compute visitable nodes
            for i in self.assembled:
                for j in range(self.total_num):
                    if self.cg.graph.has_edge(i, j) and j not in self.available:
                        self.available.append(j)
            to_remove = []
            for i in self.available:
                if i in self.assigned:
                    to_remove.append(i)
            self.available = [x for x in self.available if x not in to_remove]

            # expand the graph
            for i in self.assembled:
                if self.graph.has_node(i):
                    pass
                else:
                    self.graph.add_node(i, pos=(cg.final_center_point[i][0], cg.final_center_point[i][1]))

            for i in self.graph.nodes():
                for j in self.graph.nodes():
                    if cg.graph.has_edge(i, j) and not self.graph.has_edge(i, j):
                        self.graph.add_edge(i, j, weight=1)

            # find a path
            current_path = []
            for i in self.available:
                graph = self.graph.copy()
                graph.add_node(i, pos=(cg.final_center_point[i][0], cg.final_center_point[i][1]))
                for k in graph.nodes():
                    for j in graph.nodes():
                        if cg.graph.has_edge(k, j) and not graph.has_edge(k, j):
                            graph.add_edge(k, j, weight=1)
                p = nx.astar_path(graph, 0, i, weight='weight')
                current_path.append(p)
            if len(current_path) == 0:
                pass
            else:
                chosen = self.find_min_length_element(current_path)
                for i in range(s):
                    chosen.insert(0, 0)
                self.paths.append(chosen)
                self.assigned.append(chosen[-1])
            s+=1
            pass

def astar_run(seed=0):
    random.seed(seed)

    tic = time.time()
    cg = ConnectionGraph(37)
    cg.configFullConnection()

    # Astar benchmark
    A = Astar(1, cg, 37)
    A.run()

    for i in range(len(A.paths)):
        print(A.paths[i], len(A.paths[i]))
    print(len(A.paths))
    print(len(A.paths[-1]))
    toc = time.time()
    print(toc - tic)

    return A.paths,toc - tic


if __name__ == "__main__":
    astar_run()