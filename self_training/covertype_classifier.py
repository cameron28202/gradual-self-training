import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

class CoverTypeClassifier:
        
    def __init__(self, input_size):
        self.model = SimpleNeuralNet(input_size)
        self.scaler = StandardScaler()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = .001)
    
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


    def gradual_self_train(self, X_source_scaled, y_source, X_intermediate_scaled, y_intermediate, X_target_scaled, y_target, batch_size=5000, initial_confidence_threshold=0.6):
        self.train(X_source_scaled, y_source)

        X_train = X_source_scaled
        y_train = y_source

        num_batches = len(X_intermediate_scaled) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            X_batch = X_intermediate_scaled[start:end]
            y_batch = y_intermediate[start:end]

            proba = self.predict_proba(X_batch)
            max_proba = np.max(proba, axis=1)

            confidence_threshold = initial_confidence_threshold + (i / num_batches) * (0.9 - initial_confidence_threshold)
            confident_idx = max_proba >= confidence_threshold

            X_train = np.vstack((X_train, X_batch[confident_idx]))
            y_train = np.concatenate((y_train, self.predict(X_batch[confident_idx])))

            self.train(X_train, y_train)

            source_acc = self.evaluate(X_source_scaled, y_source)
            target_acc = self.evaluate(X_target_scaled, y_target)
            batch_acc = self.evaluate(X_batch, y_batch)
            
            print(f"Batch {i+1}/{num_batches}:")
            print(f"  Source accuracy: {source_acc:.4f}")
            print(f"  Target accuracy: {target_acc:.4f}")
            print(f"  Batch accuracy: {batch_acc:.4f}")
            print(f"  Training set size: {len(y_train)}")
            print(f"  Confidence threshold: {confidence_threshold:.2f}")
            print()
        
        return self.evaluate(X_target_scaled, y_target)

    def baseline_train(self, X_source_scaled, y_source, X_target_scaled, y_target):
        self.train(X_source_scaled, y_source)
        target_accuracy = self.evaluate(X_target_scaled, y_target)

        return target_accuracy


    def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                probabilities = outputs.numpy()
            return np.column_stack((1 - probabilities, probabilities))

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float().numpy()
        return predictions.squeeze()
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)
    
def main():
    classifier = CoverTypeClassifier(54)
    X_source_scaled, X_intermediate_scaled, X_target_scaled, y_source, y_intermediate, y_target = classifier.load_and_prepare_data()

    print("Training baseline model...")
    baseline_accuracy = classifier.baseline_train(X_source_scaled, y_source, X_target_scaled, y_target)
    print(f"Baseline target accuracy: {baseline_accuracy:.4f}")
    
    print("\nTraining with gradual self-training...")
    final_accuracy = classifier.gradual_self_train(X_source_scaled, y_source, X_intermediate_scaled, y_intermediate, X_target_scaled, y_target)
    print(f"Final target accuracy after gradual self-training: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()