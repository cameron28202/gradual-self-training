import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

class DecisionTreeNode:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.depth = depth
        self.class_probas = None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = DecisionTreeNode()
    
    def fit(self, X, y_prob):
        pass

    def grow_tree(self, X, y_prob, depth=0):
        num_samples, num_features = X.shape

        node = DecisionTreeNode(depth)

        # see if you should stop
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            # calculate avg probability for each class in leaf node
            node.class_probas = np.sum(y_prob, axis=0) / num_samples
            return node
        
        # find best split
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            # thresholds is an array of all unique values for
            # a particular feature (either x or y coordinate)

            for threshold in thresholds:

                # for each threshold, this creates a boolean
                # array (left_mask), where True means the 
                # instances feature is less than or equal to the 
                # threshold, and false means that the instance's feature
                # value is greater than the threshold. 
                left_mask = X[:, feature] <= threshold

                # we then calculate the gain, passing in the y_prob and left_mask.
                gain = self.calculate_gain(y_prob, left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
            
            if best_gain > 0:
                node.feature = best_feature
                node.threshold = best_threshold
                
                left_mask = X[:, best_feature] <= best_threshold
                
                node.left = self.grow_tree(X[left_mask], y_prob[left_mask], depth + 1)
                node.right = self.grow_tree(X[~left_mask], y_prob[~left_mask], depth + 1)

            else:
                node.class_probas = np.sum(y_prob, axis=0) / num_samples
            return node

    def gini_impurity(y_prob):
        class_probs = np.sum(y_prob, axis=0) / len(y_prob)
        return 1 - np.sum(class_probs ** 2)


    def calculate_gain(self, y_prob, left_mask):
        # total number of samples
        n = len(y_prob)

        # samples in the left tree
        n_left = np.sum(left_mask)

        # compliment (samples in the right tree)
        n_right = n - n_left

        # if either split is empty there is no gain
        if n_left == 0 or n_right == 0:
            return 0
        
        # impurity before the split
        impurity_before = self.gini_impurity(y_prob)

        # impurity after the split (left and right)
        impurity_left = self.gini_impurity(y_prob[left_mask])
        impurity_right = self.gini_impurity(y_prob[~left_mask])

        weighted_impurity_after = (n_left / n) * impurity_left + (n_right / n) * impurity_right

        return impurity_before - weighted_impurity_after

    def predict_proba(self, X):
        return np.array([self._predict_single(x) for x in X])
    
    def predict_single(self, x):
        node = self.root
        while node.left:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_probas


class SyntheticClassifier:
    def __init__(self):
        self.model = DecisionTree()

    def generate_synthetic_data(self, n_samples=100000):

        # binary classification
        n_features = 2

        # class 0 centered around -1, -1
        class_0 = np.random.randn(n_samples // 2, n_features) + [-1, -1]
        # class 1 centered around 1, 1
        class_1 = np.random.randn(n_samples // 2, n_features) + [1, 1]
        
        # stack into single 2d array, add noise
        X = np.vstack((class_0, class_1))
        X += np.random.randn(n_samples, n_features) * .5
        
        # shuffle
        random_indices = np.random.permutation(n_samples)
        X = X[random_indices]
        
        # generate probabilistic labels
        distances = np.sqrt(np.sum(X**2, axis=1))
        proba_class_1 = 1 / (1 + np.exp(-distances))
        y_prob = np.column_stack((1-proba_class_1, proba_class_1))

        return X, y_prob
    
    def load_and_prepare_data(self):
        X, y = self.generate_synthetic_data()

        # split into labeled, unlabeled and test sets
        # y unlabeled not used 
        X_source, X_temp, y_source, y_temp = train_test_split(X, y, test_size=.2, random_state=42)
        X_unlabeled, X_test, y_unlabeled, y_test = train_test_split(X_temp, y_temp, test_size=.5, random_state=42)

        # transform data
        X_source = self.scaler.fit_transform(X_source)
        X_unlabeled = self.scaler.transform(X_unlabeled)
        X_test = self.scaler.transform(X_test)

        return X_source, y_source, X_unlabeled, y_unlabeled, X_test, y_test
    
def main():
    classifier = SyntheticClassifier()

    X_source, y_source, X_unlabeled, _, X_test, y_test = classifier.load_and_prepare_data()
    

    



if __name__ == "__main__":
    main()
    
