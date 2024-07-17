import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


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
        self.root = None
    
    def fit(self, X, y_prob):
        self.num_features = X.shape[1]
        self.num_classes = y_prob.shape[1]
        self.root = self.grow_tree(X, y_prob)

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
                node.class_probas = np.mean(y_prob, axis=0)
            return node

    def gini_impurity(self, y_prob):
        return 1 - np.sum(np.mean(y_prob, axis=0) ** 2)


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
        return np.array([self.predict_single(x) for x in X])
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
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
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, n_samples=1000):

        # binary classification
        n_features = 2

        # class 0 centered around -1, -1
        class_0 = np.random.randn(n_samples // 2, n_features) + [-1, -1]
        # class 1 centered around 1, 1
        class_1 = np.random.randn(n_samples // 2, n_features) + [1, 1]
        
        # stack into single 2d array, add noise
        X = np.vstack((class_0, class_1))
        X += np.random.randn(n_samples, n_features) * .5

        # actual labels
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        # shuffle
        random_indices = np.random.permutation(n_samples)
        X = X[random_indices]
        y = y[random_indices]
        
        # generate probabilistic labels
        distances = np.sqrt(np.sum(X**2, axis=1))
        proba_class_1 = 1 / (1 + np.exp(-distances))
        y_prob = np.column_stack((1-proba_class_1, proba_class_1))

        return X, y_prob, y
    
    def load_and_prepare_data(self):
        X, y_prob, y = self.generate_synthetic_data()

        # split into labeled, unlabeled and test sets
        # y unlabeled not used 
        X_source, X_temp, y_prob_source, y_prob_temp, y_source, y_temp = train_test_split(X, y_prob, y, test_size=0.2, random_state=42)
        X_unlabeled, X_test, y_prob_unlabeled, y_prob_test, y_unlabeled, y_test = train_test_split(X_temp, y_prob_temp, y_temp, test_size=0.5, random_state=42)
        # transform data
        X_source = self.scaler.fit_transform(X_source)
        X_unlabeled = self.scaler.transform(X_unlabeled)
        X_test = self.scaler.transform(X_test)

        return X_source, y_prob_source, X_unlabeled, y_prob_unlabeled, X_test, y_prob_test, y_test
    
def main():
    classifier = SyntheticClassifier()
    X_source, y_prob_source, X_unlabeled, _, X_test, y_prob_test, y_test = classifier.load_and_prepare_data()

    classifier.model.fit(X_source, y_prob_source)

    y_pred = classifier.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Print some predictions
    print("\nSample predictions:")
    for i in range(10):
        proba = classifier.model.predict_proba(X_test[i:i+1])[0]
        pred_class = np.argmax(proba)
        true_class = y_test[i]
        print(f"Sample {i+1}: Predicted class: {pred_class}, True class: {true_class}, Probabilities: {proba}")




if __name__ == "__main__":
    main()