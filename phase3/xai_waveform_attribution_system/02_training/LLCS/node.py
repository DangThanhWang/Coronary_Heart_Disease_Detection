import numpy as np
import random
import math
from datetime import datetime

from constant import MODEL_CONSTANT
from edge import Edge

class Node(object):
    def __init__(self, input_weight, output_weight=None, target=None, name="",
                 age=1, short_term_error=0, long_term_error=0, insertion_threshold=0, inherited_error=0):
        self.input_weight = input_weight
        self.output_weight = output_weight
        self.short_term_error = short_term_error
        self.long_term_error = long_term_error
        self.insertion_threshold = insertion_threshold

        self.age = age

        self.neighbors = []

        self.activate_value = 0

        self.actual_output = None
        self.target = target

        self.inherited_error = inherited_error
        self.width_of_gaussian = 0

        self.quality_measure_for_insertion = 0

        self.name = name

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
        return node in self.get_neighbors()
    
    def delete_neighbor_by_edge(self, edge):
        self.get_neighbors().remove(edge)

        neighbor = edge.get_node()
        neighbor.get_neighbors().remove(edge)

    def delete_neighbor_by_node(self, node):
        # Remove the neighbor from this node's list of neighbors
        self.neighbors = [edge for edge in self.neighbors if edge.get_node() != node]

        # Also remove this node from the neighbor's list of neighbors if bidirectional
        node.neighbors = [edge for edge in node.neighbors if edge.get_node() != self]

    def display(self):
        print("self.insertion_threshold", self.insertion_threshold)
        print("self.long_term_error", self.long_term_error)
        print("self.inherited_error", self.inherited_error)

    def set_activate_value(self, value):
        self.activate_value = value

    def get_input_distance(self, node):
        return  np.linalg.norm(self.input_weight - node.input_weight)
    
    def get_output_distance(self, node):
        return  np.linalg.norm(self.output_weight - node.output_weight)
    
    def get_quality_measure_for_learning(self):
        return float((self.short_term_error + 1) / (self.long_term_error + 1))
    
    def get_input_learning_rate(self, best_node):
        alpha = self.get_quality_measure_for_learning() / (1 + MODEL_CONSTANT.INPUT_ADAPTATION_THRESHOLD) + self.age - 1
        learning_rate = MODEL_CONSTANT.LEARNING_RATE_FOR_BEST_NODE if best_node else MODEL_CONSTANT.LEARNING_RATE_FOR_NEIGHBOR
        if alpha <= 0:
            return 0
        return min(alpha, 1) * learning_rate
        
    def update_input_weight(self, input_node, best_node = False):
        self.input_weight = self.input_weight + self.get_input_learning_rate(best_node) * (input_node.input_weight - self.input_weight)

    def get_width_of_gaussian(self):
        return self.width_of_gaussian
    
    def update_width_of_gaussian(self):
        if (self.get_neighbor_size() == 0):
            return 0
        total_distance = 0
        for neighbor in self.get_neighbors():
            total_distance += self.get_input_distance(neighbor.get_node())
        self.width_of_gaussian = total_distance / self.get_neighbor_size()
        # return total_distance / self.get_neighbor_size()
    
    def get_output_learning_rate(self):
        alpha = self.get_quality_measure_for_learning() / (1 + MODEL_CONSTANT.OUTPUT_ADAPTATION_THRESHOLD) + self.age - 1

        if (alpha < 0):
            return 0
        return min(1, alpha) * MODEL_CONSTANT.OUTPUT_ADAPTATION_LEARNING_RATE
    
    def update_quality_measure_for_insertion(self):
        self.quality_measure_for_insertion = self.long_term_error - self.insertion_threshold * (1 + MODEL_CONSTANT.INSERTION_TOLERANCE)

    def get_local_similarity_output_weight(self):
        if (self.get_neighbor_size() == 0):
            return 0
        total_distance = 0
        for neighbor in self.get_neighbors():
            total_distance += self.get_output_distance(neighbor.get_node())
        return total_distance / self.get_neighbor_size()

    def update_error_counter(self, input_node):
        self.long_term_error = math.exp(-1 / MODEL_CONSTANT.TL) * self.long_term_error + (1 - math.exp(-1 / MODEL_CONSTANT.TL)) * np.linalg.norm(input_node.target - input_node.actual_output)
        self.short_term_error = math.exp(-1 / MODEL_CONSTANT.TS) * self.short_term_error + (1 - math.exp(-1 / MODEL_CONSTANT.TS)) * np.linalg.norm(input_node.target - input_node.actual_output)
        # print("self.long_term_error ", self.long_term_error)
        # print("self.short_term_error", self.short_term_error)

    def decrease_age_for_winner(self):
        self.age = math.exp(-1 / MODEL_CONSTANT.TY) * self.age

    # check ot again
    def decrease_insertion_threshold_for_winner(self):
        alpha = (1 + abs(self.get_quality_measure_for_learning() - 1)) / (1 + MODEL_CONSTANT.INPUT_ADAPTATION_THRESHOLD) - 1

        alpha = 0 if alpha < 0 else min(1, alpha)
        self.insertion_threshold = (1 - alpha) * math.exp(-1 / MODEL_CONSTANT.TV) * self.insertion_threshold

    def update_edge_for_winner(self, second):
        for neighbor in self.get_neighbors():
            neighbor.age += 1

            neighbor_edge = neighbor.get_node().get_neighbor_by_node(self)
            neighbor_edge.age += 1

            if (neighbor.get_node() == second):
                neighbor.age = 0
                second_edge = second.get_neighbor_by_node(self)
                second_edge.age = 0

    def update_edge(self):
        for neighbor in self.get_neighbors():
            if (neighbor.age > MODEL_CONSTANT.MAXIMUM_EDGE_AGE):
                self.delete_neighbor_by_node(neighbor.get_node())
            
        
            
        
