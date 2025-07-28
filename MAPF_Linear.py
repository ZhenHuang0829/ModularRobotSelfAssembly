import heapq
import time

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import re
import random

class TimeExpandedNetworks:
    def __init__(self, cg, T, delta_t):
        self.cg = cg
        self.tenGraph = nx.DiGraph()
        self.T = T
        self.delta_t = delta_t
        self.assembled = [0]
        self.visitable = [0]

    def build(self):
        for i in self.cg.graph.nodes:
            if self.cg.graph.has_edge(0, i):
                self.visitable.append(i)
        # add nodes
        self.tenGraph.add_node('source', pos=(0, 1))
        self.tenGraph.add_node('sink', pos=(self.T, 1))
        for t in range(self.T):
            for i in self.visitable:
                self.tenGraph.add_node(str(i) + '_' + str(t), pos=(t, -i), timestep=t, node_id=i)
                self.tenGraph.add_node(str(i) + '_' + str(t) + "'", pos=(t + 0.5, -i), timestep=t + 0.5, node_id=i)
        # add edges
        for t in range(self.T):
            for i in self.visitable:
                if t < self.T-1:
                    self.tenGraph.add_edge(str(i) + '_' + str(t), str(i) + '_' + str(t) + "'",
                                           capacity=1, flow=0, cost=0)
                    self.tenGraph.add_edge(str(i) + '_' + str(t) + "'", str(i) + '_' + str(t + 1),
                                           capacity=1, flow=0, cost=1)
                    neighbors = list(self.cg.graph.neighbors(i))
                    for n in neighbors:
                        if n in self.visitable:
                            self.tenGraph.add_node('w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                   pos=(t + 0.6, -0.5 * (i + n + 0.2)))
                            self.tenGraph.add_node('w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                   pos=(t + 0.9, -0.5 * (i + n + 0.2)))

                            self.tenGraph.add_edge('w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                   'w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                   capacity=1, flow=0, cost=1)
                            self.tenGraph.add_edge(str(i) + '_' + str(t) + "'",
                                                   'w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                   capacity=1, flow=0, cost=0)
                            self.tenGraph.add_edge(str(n) + '_' + str(t) + "'",
                                                   'w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                   capacity=1, flow=0, cost=0)
                            self.tenGraph.add_edge('w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                   str(i) + '_' + str(t + 1),
                                                   capacity=1, flow=0, cost=0)
                            self.tenGraph.add_edge('w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                   str(n) + '_' + str(t + 1),
                                                   capacity=1, flow=0, cost=0)
                else:
                    self.tenGraph.add_edge(str(i) + '_' + str(t), str(i) + '_' + str(t) + "'", capacity=1, flow=0,
                                           cost=0)
            # edges of source node
            self.tenGraph.add_edge('source', '0_0', capacity=1, flow=0, cost=0)
        # edges of sink node
        for t in range(self.T):
            for i in self.visitable:
                if i!=0:
                    self.tenGraph.add_edge(str(i) + '_' + str(t) + "'", 'sink', capacity=1, flow=0, cost=0)

    def plot(self):
        nx.draw(self.tenGraph, nx.get_node_attributes(self.tenGraph, 'pos'), with_labels=True)
        plt.show()

    def dijkstra_shortest_path(self):
        graph = self.tenGraph
        source = 'source'
        target = 'sink'
        try:
            # shortest_path = nx.dijkstra_path(graph, source, target, weight='cost')
            # shortest_path = nx.astar_path(graph, source, target, weight='cost')
            shortest_path = self.astar_path_random(graph, source, target, weight='cost')
            assert shortest_path is not None
            shortest_path_length = len(shortest_path)
            return shortest_path, shortest_path_length
        except nx.NetworkXNoPath:
            return None, float('inf')
        except AssertionError:
            return None, float('inf')

    def customized_dijkstra(self):
        graph = self.tenGraph
        start = 'source'
        target = 'sink'
        # 初始化距离字典，起点的距离为0，其它点为无穷大
        distances = {node: float('inf') for node in graph.nodes}
        distances[start] = 0

        # 前驱节点字典，用于重建最短路径
        previous_nodes = {node: None for node in graph.nodes}

        # 优先队列（最小堆），初始时只有起点
        priority_queue = [(0, start)]

        while priority_queue:
            # 获取当前距离最小的节点
            current_distance, current_node = heapq.heappop(priority_queue)

            # 如果当前节点是目标节点，结束搜索
            if current_node == target:
                break

            # 如果当前距离已经大于已记录的最短距离，则跳过（因为优先队列中的元素是可能重复的）
            if current_distance > distances[current_node]:
                continue

            # 遍历当前节点的邻居
            for neighbor, weight in graph[current_node].items():
                modified_weight = self.dynamic_weight(current_node, neighbor, weight['cost'])
                distance = current_distance + modified_weight  # 到邻居的距离

                # 如果通过当前节点能到达邻居的距离更短，更新
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        # 重建从起点到目标点的最短路径
        path = []
        node = target
        while node is not None:
            path.append(node)
            node = previous_nodes[node]

        path.reverse()

        # 如果最终路径的第一个节点是起点，则说明找到了一条路径
        if path[0] == start:
            return path, distances[target]
        else:
            return None, float('inf')  # 如果没有路径返回无穷大

    def dynamic_weight(self, current_node, neighbor, weight):
        if current_node == 'source' or neighbor == 'sink':
            return weight
        sp1 = re.split("_|'", current_node)
        sp2 = re.split("_|'", neighbor)
        if sp1[0] == sp2[0] == 'w':
            # return weight + random.uniform(0, 1)
            return weight
        elif int(sp1[1]) + 1 == int(sp2[1]):
            # return weight + random.uniform(0, 1)
            return weight
        else:
            return weight

    def astar_path_random(self, G, source, target, heuristic=None, weight='weight'):
        """
        自定义 A* 算法，当优先级相同时随机选择节点。
        输入输出与 nx.astar_path() 完全相同。

        参数:
        - G: 图（NetworkX 图对象）
        - source: 起点
        - target: 终点
        - heuristic: 启发式函数，默认为 None（相当于 Dijkstra 算法）
        - weight: 边的权重属性，默认为 'weight'

        返回:
        - path: 从起点到终点的路径（列表）
        """
        if heuristic is None:
            # 如果没有提供启发式函数，默认返回 0（相当于 Dijkstra 算法）
            def heuristic(u, v):
                return 0

        # 自定义优先级队列，支持随机选择相同优先级的节点
        class RandomPriorityQueue:
            def __init__(self):
                self.elements = []  # 存储 (priority, random_value, node) 的堆
                self.counter = 0  # 用于处理优先级相同的节点

            def put(self, node, priority):
                # 添加一个随机数作为第二优先级，确保相同优先级时随机选择
                heapq.heappush(self.elements, (priority, random.random(), node))

            def get(self):
                # 弹出优先级最高的节点
                return heapq.heappop(self.elements)[2]

            def empty(self):
                return len(self.elements) == 0

        queue = RandomPriorityQueue()  # 使用自定义的优先级队列
        queue.put(source, 0)
        came_from = {}  # 记录节点的来源
        cost_so_far = {source: 0}  # 记录从起点到当前节点的代价

        while not queue.empty():
            current = queue.get()

            if current == target:
                break  # 找到目标节点，退出循环

            for neighbor in G.neighbors(current):
                # 计算新的代价
                new_cost = cost_so_far[current] + G[current][neighbor].get(weight, 1)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, target)
                    queue.put(neighbor, priority)
                    came_from[neighbor] = current

        if len(came_from) == 0:
            raise nx.NetworkXNoPath()

        # 重建路径
        if target not in came_from:
            return None
        path = []
        current = target
        while current != source:
            path.append(current)
            current = came_from[current]
        path.append(source)
        path.reverse()

        return path

    def path_reorganize(self, path):
        path_list = []
        for i in range(len(path) - 1):
            if i % 2 == 1:
                try:
                    split_re = path[i].split('_')
                    node = split_re[0]
                    if i == 1:
                        time = split_re[1]
                        for t in range(int(time)):
                            path_list.append(0)
                    path_list.append(int(node))
                except ValueError:
                    continue
        return path_list

    def extend_ten(self, neighbors, t_final, t_nonzero):
        # new source edge
        if len(self.assembled) < self.cg.num_nodes:
            if t_nonzero >= (len(self.assembled) - 1) * self.delta_t:
                self.tenGraph.add_edge('source', '0_' + str(t_nonzero), capacity=1, flow=0, cost=0)
            else:
                self.tenGraph.add_edge('source', '0_' + str((len(self.assembled) - 1) * self.delta_t),
                                       capacity=1, flow=0, cost=0)

        for i in neighbors:
            if i not in self.visitable:
                self.visitable.append(i)
            if i not in self.assembled:
                for t in range(t_final + 1, self.T):
                    self.tenGraph.add_node(str(i) + '_' + str(t), pos=(t, -i), timestep=t, node_id=i)
                    self.tenGraph.add_node(str(i) + '_' + str(t) + "'", pos=(t + 0.5, -i), timestep=t + 0.5, node_id=i)
                for t in range(t_final + 1, self.T):
                    if t < self.T - 1:
                        self.tenGraph.add_edge(str(i) + '_' + str(t), str(i) + '_' + str(t) + "'",
                                               capacity=1, flow=0, cost=0)
                        self.tenGraph.add_edge(str(i) + '_' + str(t) + "'", str(i) + '_' + str(t + 1),
                                               capacity=1, flow=0, cost=1)
                        neighbors_1 = list(self.cg.graph.neighbors(i))
                        for n in neighbors_1:
                            if n in self.visitable:
                                self.tenGraph.add_node('w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                       pos=(t + 0.6, -0.5 * (i + n + 0.2)))
                                self.tenGraph.add_node('w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                       pos=(t + 0.9, -0.5 * (i + n + 0.2)))

                                self.tenGraph.add_edge('w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                       'w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                       capacity=1, flow=0, cost=1)
                                self.tenGraph.add_edge(str(i) + '_' + str(t) + "'",
                                                       'w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                       capacity=1, flow=0, cost=0)
                                self.tenGraph.add_edge(str(n) + '_' + str(t) + "'",
                                                       'w_' + str(t) + "_" + str(i) + "_" + str(n),
                                                       capacity=1, flow=0, cost=0)
                                self.tenGraph.add_edge('w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                       str(i) + '_' + str(t + 1),
                                                       capacity=1, flow=0, cost=0)
                                self.tenGraph.add_edge('w_' + str(t) + "'_" + str(i) + "_" + str(n),
                                                       str(n) + '_' + str(t + 1),
                                                       capacity=1, flow=0, cost=0)
                    else:
                        self.tenGraph.add_edge(str(i) + '_' + str(t), str(i) + '_' + str(t) + "'", capacity=1, flow=0,
                                               cost=0)
                    self.tenGraph.add_edge(str(i) + '_' + str(t) + "'", 'sink', capacity=1, flow=0, cost=0)
            else:
                pass

    def suppress(self, p1, p2):
        sp1 = re.split("_|'", p1)
        sp2 = re.split("_|'", p2)
        if sp1[0] == sp2[0] == 'w':
            assert sp1[1] == sp2[1]
            time = int(sp1[1])
            n1 = int(sp1[2])
            n2 = int(sp1[3])
            neighbor1 = self.cg.graph.neighbors(n1)
            neighbor2 = self.cg.graph.neighbors(n2)
            common_nodes = list(set(neighbor1) & set(neighbor2))
            for n in common_nodes:
                in_edges_to_remove = list(self.tenGraph.in_edges(str(n) + "_" + str(time + 1)))
                out_edges_to_remove = list(self.tenGraph.out_edges(str(n) + "_" + str(time) + "'"))
                common_edges = list(set(in_edges_to_remove) & set(out_edges_to_remove))
                if len(common_edges) > 0:
                    in_edges_to_remove.remove(common_edges[0])
                    out_edges_to_remove.remove(common_edges[0])
                self.tenGraph.remove_edges_from(in_edges_to_remove)
                self.tenGraph.remove_edges_from(out_edges_to_remove)

            in_edges_to_remove = list(self.tenGraph.in_edges(str(n1) + "_" + str(time + 1)))
            if ('source', str(n1) + "_" + str(time + 1)) in in_edges_to_remove:
                in_edges_to_remove.remove(('source', str(n1) + "_" + str(time + 1)))
            out_edges_to_remove = list(self.tenGraph.out_edges(str(n1) + "_" + str(time) + "'"))
            self.tenGraph.remove_edges_from(in_edges_to_remove)
            self.tenGraph.remove_edges_from(out_edges_to_remove)

            in_edges_to_remove = list(self.tenGraph.in_edges(str(n2) + "_" + str(time + 1)))
            if ('source', str(n2) + "_" + str(time + 1)) in in_edges_to_remove:
                in_edges_to_remove.remove(('source', str(n2) + "_" + str(time + 1)))
            out_edges_to_remove = list(self.tenGraph.out_edges(str(n2) + "_" + str(time) + "'"))
            self.tenGraph.remove_edges_from(in_edges_to_remove)
            self.tenGraph.remove_edges_from(out_edges_to_remove)

    def max_flow(self):
        paths = []
        paths_re = []
        while True:
            path, path_cost = self.dijkstra_shortest_path()
            if path is None:
                break
            else:
                paths.append(path)

                path_reorganized = self.path_reorganize(path)
                paths_re.append(path_reorganized)
                t_nonzero = np.nonzero(path_reorganized)[0][0]
                t_final = len(path_reorganized) - 1

                # extend
                visiting = self.tenGraph.nodes.data('node_id')[path[-2]]
                neighbors = sorted(list(self.cg.graph.neighbors(visiting)))
                self.assembled.append(visiting)
                if visiting not in self.visitable:
                    self.visitable.append(visiting)
                self.extend_ten(neighbors, t_final, t_nonzero)

                # delete
                for p in range(len(path) - 1):
                    self.tenGraph.remove_edge(path[p], path[p + 1])
                for t in range(self.T):
                    if self.tenGraph.has_edge(str(visiting) + '_' + str(t) + "'", 'sink'):
                        self.tenGraph.remove_edge(str(visiting) + '_' + str(t) + "'", 'sink')
                for p in range(len(path) - 1):
                    self.suppress(path[p], path[p + 1])
        return paths, paths_re


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
        adjacent_m = np.array([[0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
                               [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

        for i in range(self.num_nodes):
            self.graph.add_node(i, pos=(self.final_center_point[i][0], self.final_center_point[i][1]))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adjacent_m[i][j] > 0.0:
                    self.graph.add_edge(i, j, relation=adjacent_m[i][j])

    def plot(self):
        nx.draw(self.graph, self.final_center_point, with_labels=True)
        plt.show()


def mapf_demo(seed=0, T=30):
    random.seed(seed)

    start = time.time()
    cg = ConnectionGraph(13)
    cg.configFullConnection()

    ten = TimeExpandedNetworks(cg, T, 1)
    ten.build()
    paths, paths_re = ten.max_flow()

    for i in range(len(paths_re)):
        print(paths_re[i])
    print(len(paths))
    print(len(paths_re[-1]))
    print(ten.assembled)
    print(sorted(ten.assembled))
    end = time.time()
    print(end - start)

    return paths_re, end - start


if __name__ == "__main__":
    mapf_demo()
