import numpy as np
import os
import random
import re
import json
import time
from pathlib import Path

TIME_CONSTANT = 100

DATA_ROOT = Path('E:/Coronary_Heart_Disease_Detection/Data/Disease_dataset')

LABEL_PATTERN = re.compile(r'_(\d)_')


def extract_label(filename):
    match = LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot extract label from filename: {filename}")
    return int(match.group(1))

class Edge:
    def __init__(self, node, age = 1):
        self.node = node
        self.age = age

    def get_node(self):
        return self.node

class Node:
    def __init__(self, input_weight, errors=0, label=None):
        self.input_weight = input_weight  # Coordinates of the neuron
        self.errors = errors  # Error accumulated from input data
        self.label = label  # Label associated with the neuron (for classification)
        self.neighbors = []

    def get_neighbor_size(self):
        return len(self.neighbors)
    
    def get_neighbors(self):
        return self.neighbors
    
    def get_neighbor_by_node(self, other_node):
        for neighbor in self.get_neighbors():
            if (neighbor.get_node() == other_node):
                return neighbor
        return
    
    def add_neighbor(self, neighbor):
        edge = Edge(neighbor)
        self.neighbors.append(edge)

        neighbor.neighbors.append(Edge(self))

    def display_neighbor(self):
        for neighbor in self.neighbors:
            print(neighbor.get_node())

    def is_neighbor(self, node):
        for neighbor in self.get_neighbors():
            if neighbor.get_node() == node:
                return True
        return False
    
    def delete_neighbor_by_edge(self, edge):
        self.get_neighbors().remove(edge)

        neighbor = edge.get_node()

        neighbor_edge = neighbor.get_neighbor_by_node(self)
        neighbor.get_neighbors().remove(neighbor_edge)

    def delete_neighbor_by_node(self, node):
        # Remove the neighbor from this node's list of neighbors
        self.neighbors = [edge for edge in self.neighbors if edge.get_node() != node]

        # Also remove this node from the neighbor's list of neighbors if bidirectional
        node.neighbors = [edge for edge in node.neighbors if edge.get_node() != self]

    def get_input_distance(self, node):
        return  np.linalg.norm(self.input_weight - node.input_weight)
            
    def update_input_weight(self, input_node, learning_rate):
        self.input_weight = self.input_weight + learning_rate * (input_node.input_weight - self.input_weight)

class Graph(object):
    def __init__(self):
        self.graph = []

        np.random.seed(42)
        input = np.random.randint(0, 2, (30, 30))
        new_row = np.zeros((1, 30), dtype=int)
        updated_input_array = np.append(input, new_row, axis=0)
        random_node_1 = Node(input_weight=updated_input_array)
        self.graph.append(random_node_1)

        input = np.random.randint(0, 2, (30, 30))
        new_row = np.zeros((1, 30), dtype=int)
        updated_input_array = np.append(input, new_row, axis=0)
        random_node_2 = Node(input_weight=updated_input_array)
        self.graph.append(random_node_2)

        random_node_1.add_neighbor(random_node_2)

    def add_node(self, node):
        """Adds a new node to the graph."""
        # node = Node(value)
        if node not in self.graph:
            self.graph.append(node)

    def remove_node(self, node):
        self.graph.remove(node)
        for neighbor in node.get_neighbors():
            neighbor_node = neighbor.get_node()
            for t in neighbor_node.get_neighbors():
                if (t.get_node() == node):
                    neighbor_node.get_neighbors().remove(t)

    def get_edges_number(self):
        count = 0
        for node in self.graph:
            count += node.get_neighbor_size()
        return round(count / 2)
    
    def find_best_and_second_best_node(self, input_node):
        # Initialize variables to store the best and second-best nodes and their distances
        best_node = None
        second_best_node = None
        best_distance = float('inf')
        second_best_distance = float('inf')

        # Iterate over all nodes in the graph
        for node in self.graph:
            if node == input_node:
                continue  # Skip comparing the node with itself
            
            # Calculate the distance between the input node and the current node
            distance = input_node.get_input_distance(node)
            
            # Update the best and second-best nodes based on the distance
            if distance <= best_distance:
                second_best_distance = best_distance
                second_best_node = best_node
                best_distance = distance
                best_node = node
            elif distance <= second_best_distance:
                second_best_distance = distance
                second_best_node = node

        return best_node, second_best_node

class GNG:
    def __init__(self, dataset_path, dataset_name="DEFAULT", max_nodes=1000, winner_lr=0.1, second_lr=0.003, max_edge = 50, theta = 30):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.max_nodes = max_nodes
        self.graph = Graph()

        self.winner_lr = winner_lr
        self.second_lr = second_lr
        self.max_edge = max_edge
        self.step = 0
        self.theta = theta
        self.alpha = 0.5

        self.train_error = 0
        self.success_samples = 0

        self.log = []
        self.count_insert = 0

    def fit(self, max_epoch):
        for i in range(max_epoch):
            samples_list = []
            for filename in os.listdir(self.dataset_path):
                if (filename.endswith(".npy")):
                    samples_list.append(filename)
            random.seed(42)
            random.shuffle(samples_list)

            for filename in samples_list:
                self.step += 1
                file_path = os.path.join(self.dataset_path, filename)
                ecg_array = np.load(file_path)

                label = extract_label(filename)

                input_node = Node(ecg_array, label=label)
                first, second = self.graph.find_best_and_second_best_node(input_node)

                if (label == first.label):
                    self.success_samples += 1
                first.label = label

                for neighbor in first.get_neighbors():
                    neighbor.age += 1
                    neighbor_edge = neighbor.get_node().get_neighbor_by_node(first)
                    neighbor_edge.age += 1

                if (first.is_neighbor(second)):
                        first_edge = first.get_neighbor_by_node(second)
                        first_edge.age = 0
                        second_edge = second.get_neighbor_by_node(first)
                        second_edge.age = 0
                else:
                    first.add_neighbor(second)


                first.errors = first.errors + first.get_input_distance(input_node)

                first.update_input_weight(input_node, self.winner_lr)
                for neighbor in first.get_neighbors():
                    neighbor.get_node().update_input_weight(input_node, self.second_lr)

                if (self.step % self.theta == 0):
                    self.count_insert += 1
                    max_error = 0
                    max_node = None
                    count_edge = 0
                    for node in self.graph.graph:
                        count_edge += len(node.neighbors)

                        if (max_error < node.errors):
                            max_error = max(max_error, node.errors)
                            max_node = node
                        for neighbor in node.get_neighbors():
                            if (neighbor.age > self.max_edge):
                                node.delete_neighbor_by_edge(neighbor)
                        if (node.get_neighbor_size() < 2 and len(self.graph.graph) > 2):
                            self.graph.remove_node(node)

                    max_error = float('-inf')
                    second_node = None
                    for neighbor in max_node.get_neighbors():
                        neighbor_node = neighbor.get_node()
                        if (max_error < neighbor_node.errors):
                            max_error = max(max_error, neighbor_node.errors)
                            second_node = neighbor_node
                    if (second_node):
                        new_node = Node(input_weight=(max_node.input_weight + second_node.input_weight) / 2, 
                                        label=max_node.label,
                                        errors=(max_node.errors + second_node.errors)/2)                    
                        self.graph.add_node(new_node)
                        new_node.add_neighbor(max_node)
                        new_node.add_neighbor(second_node)
                        max_node.delete_neighbor_by_node(second_node)

                        max_node.errors = self.alpha * max_node.errors
                        second_node.errors = self.alpha * second_node.errors
                
                if (self.step % TIME_CONSTANT == 0):
                    self.log.append({
                        "step": self.step,
                        "success_rate": self.success_samples / self.step,
                        "num_nodes": len(self.graph.graph),
                        "num_edges": count_edge / 2
                    })
            print("Dataset name: ", self.dataset_name, " - Epoch ", i + 1, "out of ", max_epoch)
        with open("./" + str(self.step) + ".json", 'w') as json_file:
            json.dump(self.log, json_file)
            self.log = []

    def eval(self, eval_dataset):
        print("Start Evaluate GNG")
        count = 0
        count_rn = 0
        count_ra = 0
        count_wn = 0
        count_wa = 0
        datasetPath = eval_dataset
        count_true = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0
        }

        index = 0
        for filename in os.listdir(datasetPath):
            if (filename.endswith(".npy")):
                index += 1
                file_path = os.path.join(datasetPath, filename)
                ecg_array = np.load(file_path)

                label = extract_label(filename)
                count_true[str(label)] += 1

                input_node = Node(ecg_array, label=label)
                first, second = self.graph.find_best_and_second_best_node(input_node)

                if (first.label == label):
                    count += 1
                    if (label == 0):
                        count_rn += 1
                    elif (label == 1):
                        count_ra += 1
                    elif (label == 2):
                        count_wn += 1
                    elif (label == 3):
                        count_wa += 1

        # print("RN ", count_rn, " out of ", count_true["0"])
        # print("RA ", count_ra, " out of ", count_true["1"])
        # print("WN ", count_wn, " out of ", count_true["2"])
        # print("WA ", count_wa, " out of ", count_true["1"])

        total_samples = count_true["0"] + count_true["1"] + count_true["2"] + count_true["3"]
        accuracy = (count_rn + count_ra + count_wn + count_wa) / total_samples * 100 if total_samples else 0
        print("Accuracy: ", round(accuracy, 2), "%")
        tar_den = count_true["1"] + count_true["3"]
        tar = (count_ra + count_wa) / tar_den * 100 if tar_den else 0
        print("True Acceptance Rate (TAR): ", round(tar, 2), "%")
        far_den = count_true["0"] + count_true["2"]
        far = ((count_true["0"] - count_rn) + (count_true["2"] - count_wn)) / far_den * 100 if far_den else 0
        print("False Acceptance Rate (FAR): ", round(far, 2), "%")
        print("Number of nodes: ", len(self.graph.graph))


count_insert = 0
dataset_path = str(DATA_ROOT / 'Env1' / 'NumpyData')
model = GNG(dataset_path=dataset_path, dataset_name="ENV1", max_nodes=1000)
model.fit(max_epoch=5)

# Change env
model.dataset_name = "ENV2"
model.dataset_path = str(DATA_ROOT / 'Env2' / 'NumpyData')
model.fit(max_epoch=5)

# Change env
model.dataset_name = "ENV3"
model.dataset_path = str(DATA_ROOT / 'Env3' / 'NumpyData')
model.fit(max_epoch=7)

# Change env
model.dataset_name = "ENV4"
model.dataset_path = str(DATA_ROOT / 'Env4' / 'NumpyData')
model.fit(max_epoch=7)


# Eval model 
start_time = time.time()
model.eval(str(DATA_ROOT / 'Eval' / 'NumpyData'))
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time for: {execution_time:.2f} seconds")
print("Number of edges: ", model.graph.get_edges_number())



