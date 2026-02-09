import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import random
import re
import matplotlib.pyplot as plt
import networkx as nx
import time


from graph import Graph
# from new_graph import Graph
from node import Node
from constant import MODEL_CONSTANT, DATASET

LABEL_PATTERN = re.compile(r'_(\d)_')


def extract_label(filename):
    match = LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot extract label from filename: {filename}")
    return int(match.group(1))


def softmax(array):
    # Subtract the maximum value from the array for numerical stability
    exp_values = np.exp(array - np.max(array))
    softmax_value = exp_values / np.sum(exp_values)

    max_value = np.max(softmax_value)

    # Replace maximum with 1 and others with 0
    softmax_value = np.where(softmax_value == max_value, 1, 0)

    return softmax_value.reshape(1, -1)

class LLCS(object):
    def __init__(self, dataset_log_path, dataset_name="DEFAULT"):
        self.graph = Graph()
        self.dataset_name = dataset_name
        self.step = 0

        self.total_mse = 0

        self.true_detect = 0

        # Current timestamp
        timestamp_str = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        self.dataset_log_path = dataset_log_path + timestamp_str + "/"
        os.makedirs(self.dataset_log_path , exist_ok=True)

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_log_path(self, dataset_log_path):
        # Current timestamp
        timestamp_str = timestamp_str = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        self.dataset_log_path = dataset_log_path + timestamp_str + "/"
        os.makedirs(self.dataset_log_path , exist_ok=True)

    def export_log(self):

        with open(self.dataset_log_path + "log_until_" + str(self.step) + ".json", 'w') as json_file:
            json.dump(self.graph.capture_data, json_file, indent=4)  # indent=4 for pretty-printing
            
        self.graph.capture_data = []

    def export_parameter(self):
        model_constants = {attr: value for attr, value in MODEL_CONSTANT.__dict__.items() if not attr.startswith('__')}

        with open(self.dataset_log_path + "parameters_log.txt", 'w') as json_file:
            json.dump(model_constants, json_file, indent=4)  # indent=4 for pretty-printing

    def export_graph(self):
        # Visualize the graph with 2D layout
        # plt.figure(figsize=(12, 8))
        # # node_positions = nx.spring_layout(self.graph.new_graph, seed=42)  # Position nodes in 2D space using spring layout

        # # Plot the nodes, edges, and labels
        # nx.draw_networkx_nodes(self.graph.new_graph, self.graph.node_positions, node_size=500, node_color='skyblue', alpha=0.7)
        # nx.draw_networkx_edges(self.graph.new_graph, self.graph.node_positions, width=1.0, alpha=0.5, edge_color='gray')
        # nx.draw_networkx_labels(self.graph.new_graph, self.graph.node_positions, font_size=10, font_color='black')
        # # nx.draw(self.graph.new_graph, pos=self.node_positions, with_labels=True, node_color='skyblue', edge_color='gray')

        # # Set plot title and labels
        # plt.title('Step ' + str(self.step))
        # plt.axis('off')  # Turn off the axis for better visualization

        os.makedirs(self.dataset_log_path + "graph/", exist_ok=True)
        nx.write_graphml(G, self.dataset_log_path + "graph/" + str(self.step) + ".graphml")

    def capture_model(self):
        data = {
            "step": self.step,
            "num_nodes": self.graph.get_graph_size(),
            "num_edges": self.graph.get_graph_edge(),
            "mse": self.total_mse / self.step,
            "success_rate": self.true_detect / self.step
        }
        self.graph.capture_data.append(data)

        if (self.step % 1000 == 1):
            self.export_log()


    def fit(self, max_epoch=1):
        self.export_parameter()
        dataset_path = DATASET[self.dataset_name]
        # dataset_path = "./Disease_dataset/Raw/Dataset3_Official/NumpyData_3031/train/"

        t_ins = 0
        for i in range(max_epoch):
            print("Dataset name: ", self.dataset_name, " - EPOCH ", i + 1, " out of ", max_epoch)
            samples_list = []
            for filename in os.listdir(dataset_path):
                if (filename.endswith(".npy")):
                    samples_list.append(filename)
            random.seed(42)
            random.shuffle(samples_list)

            for filename in samples_list:
                file_path = os.path.join(dataset_path, filename)
                ecg_array = np.load(file_path)

                label = extract_label(filename)
                
                output = np.zeros((1, 4), dtype=int)
                if 0 <= label < 4:
                    # Set the specified index to 1
                    output[0, label] = 1
                else:
                    print("Invalid index. Please provide a value between 0 and 5.")
            
                input_node = Node(ecg_array, target=output)

                # 1.1. find best and second match
                first, second = model.graph.find_best_and_second_best_node(input_node)

                # update quality measure for learning
                first.update_input_weight(input_node, best_node=True)
                for neighbor in first.get_neighbors():
                    neighbor.get_node().update_input_weight(input_node, best_node=False)

                self.graph.activate_node(input_node)
                self.graph.update_out_weight(input_node)

                self.step += 1
                self.total_mse += np.mean((input_node.actual_output - input_node.target) ** 2)

                if (np.all(softmax(input_node.actual_output) == input_node.target)):
                    self.true_detect += 1

                t_ins += 1
                
                if (t_ins == MODEL_CONSTANT.THETA * self.graph.get_graph_size()):
                    self.graph.log[self.step] = "insert"
                    self.graph.update_BI()
                    self.graph.get_q_and_f_node(self.step)
                    t_ins = 0

                self.graph.check_deletion_criteria()

                first.update_error_counter(input_node)
                first.decrease_age_for_winner()
                first.decrease_insertion_threshold_for_winner()

                if (not first.is_neighbor(second)):
                    first.add_neighbor(second)

                first.update_edge_for_winner(second)

                if (self.graph.get_graph_size() > 2):
                    self.graph.update_node() 

                if (self.step % MODEL_CONSTANT.CAPTURE_TIME == 0):
                    self.capture_model()
                # if (self.step % 200) == 1:
                #     self.export_graph()
            if (i == max_epoch - 1):
                self.export_log()

    def evaluate(self):
        print("Start Evaluate LLCS")
        count = 0
        count_rn = 0
        count_ra = 0
        count_wn = 0
        count_wa = 0
        datasetPath = DATASET.get("Eval")

        if datasetPath is None:
            raise ValueError("Eval dataset path is not configured in DATASET")

        count_true = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
        }

        index = 0
        for filename in os.listdir(datasetPath):
            if (filename.endswith(".npy")):
                index += 1
                file_path = os.path.join(datasetPath, filename)
                ecg_array = np.load(file_path)

                label = extract_label(filename)
                if label not in count_true:
                    continue
                count_true[label] += 1

                output = np.zeros((1, 4), dtype=int)
                if 0 <= label < 4:
                    # Set the specified index to 1
                    output[0, label] = 1
                else:
                    print("Invalid index. Please provide a value between 0 and 5.")

            
                input_node = Node(ecg_array, target=output)
                model.graph.activate_node(input_node=input_node)

                output = model.graph.get_actual_output()

                if (np.all(softmax(output) == input_node.target)):
                    count += 1
                    if (label == 0):
                        count_rn += 1
                    elif (label == 1):
                        count_ra += 1
                    elif (label == 2):
                        count_wn += 1
                    elif (label == 3):
                        count_wa += 1

        # print("Total ",count, "out of 1550")
        # print("RN ", count_rn)
        # print("RA ", count_ra)
        # print("WN ", count_wn)
        # print("WA ", count_wa)

        total_samples = count_true[0] + count_true[1] + count_true[2] + count_true[3]
        accuracy = (count_rn + count_ra + count_wn + count_wa) / total_samples * 100 if total_samples else 0
        print("Accuracy: ", round(accuracy, 2), "%")
        tar_den = count_true[1] + count_true[3]
        tar = (count_ra + count_wa) / tar_den * 100 if tar_den else 0
        print("True Acceptance Rate (TAR): ", round(tar, 2), "%")
        far_den = count_true[0] + count_true[2]
        far = ((count_true[0] - count_rn) + (count_true[2] - count_wn)) / far_den * 100 if far_den else 0
        print("False Acceptance Rate (FAR): ", round(far, 2), "%")
        print("Number of nodes: ", len(self.graph.graph))


    def numpy_log(self):
        timestamp_str = timestamp_str = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        output_path = "./Node_log/"

        log_path = output_path + timestamp_str + "/"
        os.makedirs(log_path , exist_ok=True)

        index = 0
        for node in self.graph.graph:
            np.save(log_path + str(index) + ".npy", node.input_weight)
            index += 1


log_name = "Env_Change"
dataset_log_path = "./Thesis_log/" + log_name + "/"

dataset_name = "Env1"
model = LLCS(dataset_name=dataset_name, dataset_log_path=dataset_log_path)
model.fit(max_epoch = 5)
model.dataset_name = "Env2"
model.fit(max_epoch = 5)
model.dataset_name = "Env3"
model.fit(max_epoch = 7)
model.dataset_name = "Env4"
model.fit(max_epoch = 7)


start_time = time.time()
model.evaluate()
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time for: {execution_time:.2f} seconds")
print("Number of edges: ", model.graph.get_graph_edge())

