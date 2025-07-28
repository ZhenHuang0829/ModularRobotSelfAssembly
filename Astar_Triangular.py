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
            [[0, 0], [0, h], [0, 2*h],[0, 3*h],[0, 4*h],[0, 5*h],[0, 6*h],
             [0, -h], [0, -2*h],[0, -3*h],[0, -4*h],[0, -5*h],[0, -6*h]])

    def configFullConnection(self):
        adjacent_m = np.array([[0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [4, 0, 1, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 4, 0, 1, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 4, 0, 1, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                               [3, 2, 0, 0, 0, 0, 1, 0, 0, 6, 0, 0, 0, 0, 0],
                               [0, 3, 2, 0, 0, 4, 0, 1, 0, 5, 6, 0, 0, 0, 0],
                               [0, 0, 3, 2, 0, 0, 4, 0, 1, 0, 5, 6, 0, 0, 0],
                               [0, 0, 0, 3, 2, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0],
                               [0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 1, 0, 6, 0, 0],
                               [0, 0, 0, 0, 0, 0, 3, 2, 0, 4, 0, 1, 5, 6, 0],
                               [0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 4, 0, 0, 5, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 1, 6],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 4, 0, 5],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0]])

        for i in range(self.num_nodes):
            self.graph.add_node(i)
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

        s = 0
        while True:
            if len(self.assembled) == self.total_num:
                break

            # 计算已组装结点
            if len(self.paths) != 0:
                for p in self.paths:
                    if len(p) == s+1:
                        self.assembled.append(p[-1])

            # 计算可被选择的结点
            for i in self.assembled:
                for j in range(self.total_num):
                    if self.cg.graph.has_edge(i, j) and j not in self.available:
                        self.available.append(j)
            to_remove = []
            for i in self.available:
                if i in self.assigned:
                    to_remove.append(i)
            self.available = [x for x in self.available if x not in to_remove]

            # 扩展图
            for i in self.assembled:
                if self.graph.has_node(i):
                    pass
                else:
                    self.graph.add_node(i)

            for i in self.graph.nodes():
                for j in self.graph.nodes():
                    if self.cg.graph.has_edge(i, j) and not self.graph.has_edge(i, j):
                        self.graph.add_edge(i, j, weight=1)


            current_path = []
            for i in self.available:
                graph = self.graph.copy()
                graph.add_node(i)
                for k in graph.nodes():
                    for j in graph.nodes():
                        if self.cg.graph.has_edge(k, j) and not graph.has_edge(k, j):
                            graph.add_edge(k, j, weight=1)
                p = nx.astar_path(graph, 0, i, weight='weight')
                # p_len = nx.astar_path_length(self.graph, 0, self.available[i], weight='weight')
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
    cg = ConnectionGraph(15)
    cg.configFullConnection()

    A = Astar(1, cg, 15)
    A.run()

    for i in range(len(A.paths)):
        print(A.paths[i])
    print(len(A.paths))
    print(len(A.paths[-1]))
    print(A.assembled)
    print(sorted(A.assembled))
    toc = time.time()
    print(toc - tic)

    return A.paths,toc - tic


if __name__ == "__main__":
    astar_run()