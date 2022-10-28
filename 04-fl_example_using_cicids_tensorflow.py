import os
import pandas as pd
import numpy as np

from collections import OrderedDict
from typing import List, Tuple

# We load in the data generated from notebook two of this series
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print ("Data loaded...")
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

NUM_OF_CLIENTS = 10
NUM_OF_ROUNDS = 5

fl_X_train = []
fl_y_train = []
METHODS = ['stratified', 'split_by_attack', 'split_by_count'] 
METHOD = METHODS[0]

if METHOD == 'stratified':
    ## 1. STRATIFIED SAMPLING
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=NUM_OF_CLIENTS, shuffle=True, random_state=42)
    skf.get_n_splits(X_train, y_train)
    for train_index, test_index in skf.split(X_train, y_train):
        fl_X_train.append(X_train[test_index])
        fl_y_train.append(y_train[test_index]) 
elif METHOD == 'split_by_attack':
    for i in np.unique(y_test):
        print ("Get class: ", i)
        indices = np.where(y_train==i)
        print ("Shape of class ", i , " : ", X_train[indices].shape)
        fl_X_train.append(X_train[indices])
        fl_y_train.append(y_train[indices]) 
elif METHOD == 'split_by_count':
    COUNT = 10000
    s = np.arange(0,X_train.shape[0],COUNT)
    for i in range(len(s)-1):
        fl_X_train.append(X_train[s[i]:s[i+1]])
        fl_y_train.append(y_train[s[i]:s[i+1]])
# CHECK IF THIS REMAINS THE SAME OR CHANGED
NUM_OF_CLIENTS = len(fl_X_train)
print ("NUM_OF_CLIENTS:", NUM_OF_CLIENTS)


print ("Checking data split groups")
for i in range(len(fl_X_train)):
    print (i, ':', "X shape", fl_X_train[i].shape, " Y shape:" , fl_y_train[i].shape)

print ("Importing Federated Learning environment...")
import flwr as fl
import numpy as np
import tensorflow as tf
print("flwr", fl.__version__)
print("numpy", np.__version__)
print("tf", tf.__version__)
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

class NumpyFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_data, train_labels):
        self.model = model
        self.cid = cid
        self.train_data = train_data
        self.train_labels = train_labels

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        print ("Client ", self.cid, "Training...")
        self.model.fit(self.train_data, self.train_labels, epochs=1, batch_size=32)
        print ("Client ", self.cid, "Training complete...")
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        print ("Client ", self.cid, "Evaluating...")
        loss, accuracy = self.model.evaluate(self.train_data, y_test)
        print ("Client ", self.cid, "Evaluating complete...", accuracy, loss)
        return loss, len(self.train_data), {"accuracy": accuracy}


def client_fn(cid: str) -> NumpyFlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    print ("Client ID:", cid)

    model = Sequential([
      #Flatten(input_shape=(79,1)),
      Flatten(input_shape=(fl_X_train[0].shape[1] , 1)),
      Dense(256, activation='sigmoid'),
      Dense(128, activation='sigmoid'), 
      Dense(18, activation='sigmoid'),  
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    partition_id = int(cid)
    X_train_c = fl_X_train[partition_id]
    y_train_c = fl_y_train[partition_id]

    # Create a  single Flower client representing a single organization
    return NumpyFlowerClient(cid, model, X_train_c, y_train_c)


print ("Deploy simulation...")
# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2, #10,
        min_evaluate_clients=2, #5,
        min_available_clients=2, #10,
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_OF_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_OF_ROUNDS),
    strategy=strategy,
)