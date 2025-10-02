import numpy as np
import os
import re
from collections import Counter
import random
import json
import time


TIME_CONSTANT = 100

# Node class for the SOM
class Node:
    def __init__(self, dim, learning_rate=0.1, radius=1):
        self.weights = np.random.rand(dim)  # Initialize the weights randomly
        self.learning_rate = learning_rate  # Learning rate for the update
        self.radius = radius  # Neighborhood radius for the update
        self.label = None  # No label initially

    def update_weights(self, input_data, distance_to_bmu, current_learning_rate):
        """
        Update the weights of the node based on the input data and its distance to the BMU.
        """
        influence = np.exp(-distance_to_bmu / (2 * (self.radius ** 2)))  # Influence based on distance
        self.weights += current_learning_rate * influence * (input_data - self.weights)

    def get_distance(self, input_data):
        """
        Calculate the Euclidean distance between the node's weights and the input data.
        Ensure input_data has the correct shape (flattened if necessary).
        """
        return np.linalg.norm(self.weights - input_data)

# KSOM class for the Self-Organizing Map
class KSOM:
    def __init__(self, grid_size, dim, dataset_name = "DEFAULT", learning_rate=0.1, radius=1, max_iter=100):
        self.grid_size = grid_size
        self.dim = dim  # Dimensionality of the input data
        self.learning_rate = learning_rate  # Initial learning rate
        self.radius = radius  # Neighborhood radius
        self.max_iter = max_iter  # Number of iterations
        self.nodes = np.array([[Node(dim, learning_rate, radius) for _ in range(grid_size)] for _ in range(grid_size)])
        self.train_error = 0
        self.true_detect = 0
        self.step = 1
        self.log = []
        self.log_path = './data_log.json'
        self.mse = 0

        self.dataset_name = dataset_name



    def train(self, data, labels):
        """
        Train the Kohonen map using the input data and labels.
        """
        
        for epoch in range(self.max_iter):
            # np.random.shuffle(data)  # Shuffle data to ensure better learning
            for input_data, label in zip(data, labels):
                self.step += 1
                # Find the best matching unit (BMU)

                input_data = input_data.flatten()
                bmu, bmu_pos = self.find_bmu(input_data)

                # Update weights of the BMU and its neighbors
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        node = self.nodes[i, j]
                        distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_pos))
                        node.update_weights(input_data, distance_to_bmu, self.learning_rate)

                # Assign label based on majority class of the BMU's neighborhood
                if (label == bmu.label):
                    self.true_detect += 1
                self.update_labels(bmu, label)
                
                if (self.step % TIME_CONSTANT == 0):
                    self.log.append({
                        "step": self.step,
                        "success_rate": self.true_detect / self.step
                    })

            # Decay learning rate and radius over time
            self.learning_rate = self.learning_rate * (1 - epoch / self.max_iter)
            self.radius = self.radius * (1 - epoch / self.max_iter)
            print("Dataset name: ", self.dataset_name, " - EPOCH ", epoch + 1, " out of ", self.max_iter)

        with open("./" + str(self.step) + ".json", 'w') as json_file:
            json.dump(self.log, json_file)

    def find_bmu(self, input_data):
        """
        Find the Best Matching Unit (BMU) for the input data.
        """
        min_dist = float('inf')
        bmu = None
        bmu_pos = None
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node = self.nodes[i, j]
                dist = node.get_distance(input_data)
                if dist < min_dist:
                    min_dist = dist
                    bmu = node
                    bmu_pos = (i, j)
        return bmu, bmu_pos

    def update_labels(self, bmu, label):
        """
        Update the label of the BMU based on the majority class of its neighborhood.
        """
        bmu.label = label  # Assign the label to the BMU

    def predict(self, input_data):
        """
        Predict the class for a given input data point by finding the BMU.
        """
        bmu, _ = self.find_bmu(input_data)
        return bmu.label

    def get_map(self):
        """
        Return the weight map of the SOM.
        """
        return np.array([[node.weights for node in row] for row in self.nodes])

def load_data_from_directory(directory_path, seed=42):
    data = []
    labels = []
    
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)  # Seed for random.shuffle
        np.random.seed(seed)  # Seed for numpy random operations
    
    # List all files in the directory
    file_list = [filename for filename in os.listdir(directory_path) if filename.endswith('.npy')]
    
    # Shuffle the list of files randomly
    random.shuffle(file_list)
    
    # Loop through the shuffled list of files
    for filename in file_list:
        # Load the data
        file_path = os.path.join(directory_path, filename)
        file_data = np.load(file_path)
        
        # Extract label from the filename (e.g., 2024-11-02T09:09:35.743Z_0_Resting-Normal.npy)
        label_match = re.search(r'_(\d)_', filename)
        label = int(label_match.group(1)) if label_match else None
        
        # Add the data and label to the respective lists
        # for sample in file_data:
        data.append(file_data)  # Features
        labels.append(label)  # Corresponding label

    return np.array(data), np.array(labels)

if __name__ == "__main__":    
    data, labels = load_data_from_directory("E:/Coronary_Heart_Disease_Detection/Data/Disease_dataset/Env1/NumpyData/")
    # Create and train the KSOM model
    som = KSOM(grid_size=10, dim=30*31, learning_rate=0.1, radius=1, max_iter=5, dataset_name="ENV1")
    som.train(data, labels)

    data, labels = load_data_from_directory("E:/Coronary_Heart_Disease_Detection/Data/Disease_dataset/Env2/NumpyData/")
    som.dataset_name = "ENV2"
    som.max_iter = 5
    som.train(data, labels)

    som.dataset_name = "ENV3"
    som.max_iter = 7
    data, labels = load_data_from_directory("E:/Coronary_Heart_Disease_Detection/Data/Disease_dataset/Env3/NumpyData/")
    som.train(data, labels)

    som.dataset_name = "ENV4"
    som.max_iter = 7
    data, labels = load_data_from_directory("E:/Coronary_Heart_Disease_Detection/Data/Disease_dataset/Env4/NumpyData/")
    som.train(data, labels)
    
    data, labels = load_data_from_directory("E:/Coronary_Heart_Disease_Detection/Data/Disease_dataset/Eval/NumpyData/")
    count = 0
    count_rn = 0
    count_ra = 0
    count_wn = 0
    count_wa = 0
    count_true = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0
    }

    print("Start Evaluate KSOM")


    start_time = time.time()

    for input_data, label in zip(data, labels):
        input_data = input_data.flatten()
        bmu, bmu_pos = som.find_bmu(input_data)
        count_true[str(label)] += 1

        if (label == bmu.label):
            if (label == 0):
                count_rn += 1
            elif (label == 1):
                count_ra += 1
            elif (label == 2):
                count_wn += 1
            elif (label == 3):
                count_wa += 1

    # print("RN ", count_rn)
    # print("RA ", count_ra)
    # print("WN ", count_wn)
    # print("WA ", count_wa)
    total_samples = count_true["0"] + count_true["1"] + count_true["2"] + count_true["3"]
    accuracy = (count_rn + count_ra + count_wn + count_wa) / total_samples * 100 if total_samples else 0
    print("Accuracy: ", round(accuracy, 2), "%")
    tar_den = count_true["1"] + count_true["3"]
    tar = (count_ra + count_wa) / tar_den * 100 if tar_den else 0
    print("True Acceptance Rate (TAR): ", round(tar, 2), "%")
    far_den = count_true["0"] + count_true["2"]
    far = ((count_true["0"] - count_rn) + (count_true["2"] - count_wn)) / far_den * 100 if far_den else 0
    print("False Acceptance Rate (FAR): ", round(far, 2), "%")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for: {execution_time:.2f} seconds")
