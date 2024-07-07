import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class CoverTypeClassifier:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):

        # fetch dataset (x is feature data, y is labels)
        X, y = fetch_covtype(return_X_y=True)

        # change to binary classification
        # either spruce/fir or not
        y = (y == 1).astype(int)

        # index of the distance to water in the cover type dataset
        water_dist_index = 3

        # sort datasets based on distance to water
        sorted_indices = np.argsort(X[:, water_dist_index])

        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # as defined in paper:
        #  source dataset has 50,000 elements
        #  intermediate dataset is everything between (~400,000)
        #  target dataset has 50,000 elements
        total_samples = X_sorted.shape[0]
        source_size = 50000
        target_size = 50000
        intermediate_size = total_samples - source_size - target_size

        # define source, intermdiate and target datasets
        X_source = X_sorted[:source_size]
        y_source = y_sorted[:source_size]

        X_intermediate = X_sorted[source_size:intermediate_size+source_size]
        y_intermediate = y_sorted[source_size:intermediate_size+source_size]

        X_target = X_sorted[-target_size:]
        y_target = y_sorted[-target_size:]

        # scale data
        self.scaler.fit(X_source)
        X_source_scaled = self.scaler.transform(X_source)
        X_intermediate_scaled = self.scaler.transform(X_intermediate)
        X_target_scaled = self.scaler.transform(X_target)

        return X_source_scaled, X_intermediate_scaled, X_target_scaled, y_source, y_intermediate, y_target


    def gradual_self_train(self, X_source_scaled, y_source, X_intermediate_scaled, y_intermediate, X_target_scaled, y_target, batch_size= 1000, confidence_threshold=0.8):
        # train on initial labeled dataset
        self.train(X_source_scaled, y_source)

        X_train = X_source_scaled
        y_train = y_source

        num_batches = len(X_intermediate_scaled) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            # create x and y batch
            X_batch = X_intermediate_scaled[start:end]
            y_batch = y_intermediate[start:end]

            # give psuedo labels to unlabeled dataset
            proba = self.predict_proba(X_batch)

            max_proba = np.max(proba, axis=1)

            # add labels that are above passsed in confidence threshold (default .8)
            confident_idx = max_proba >= confidence_threshold

            X_train = np.vstack((X_train, X_batch[confident_idx]))
            y_train = np.concatenate((y_train, self.predict(X_batch[confident_idx])))

            # train on new psuedo labeled data
            self.train(X_train, y_train)

            # evaluate performance
            source_acc = self.evaluate(X_source_scaled, y_source)
            target_acc = self.evaluate(X_target_scaled, y_target)
            batch_acc = self.evaluate(X_batch, y_batch)
            
            print(f"Batch {i+1}/{num_batches}:")
            print(f"  Source accuracy: {source_acc:.4f}")
            print(f"  Target accuracy: {target_acc:.4f}")
            print(f"  Batch accuracy: {batch_acc:.4f}")
            print(f"  Training set size: {len(y_train)}")
            print()
        
        return self.evaluate(X_target_scaled, y_target)

    def baseline_train(self, X_source_scaled, y_source, X_target_scaled, y_target):
        self.train(X_source_scaled, y_source)
        target_accuracy = self.evaluate(X_target_scaled, y_target)

        return target_accuracy


    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)