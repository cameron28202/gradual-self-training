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
        self.model = None
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, n_samples=50000):

        # binary classification
        n_features = 2

        # class 0 centered around -1, -1
        class_0 = np.random.randn(n_samples // 2, n_features) + [-1, -1]
        # class 1 centered around 1, 1
        class_1 = np.random.randn(n_samples // 2, n_features) + [1, 1]
        
        # stack into single 2d array
        X = np.vstack((class_0, class_1))

        #actual labels
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

        X += np.random.randn(n_samples, n_features) * .5
        
        # shuffle
        random_indices = np.random.permutation(n_samples)
        X = X[random_indices]
        y = y[random_indices]
        
        class_0_center = np.array([-1, -1])
        class_1_center = np.array([1, 1])
        
        distances_0 = np.linalg.norm(X - class_0_center, axis=1)
        distances_1 = np.linalg.norm(X - class_1_center, axis=1)
        
        proba_class_1 = distances_0 / (distances_0 + distances_1)
        y_prob = np.column_stack((1 - proba_class_1, proba_class_1))
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
    
    def gradual_self_train(self, X_source, y_prob_source, X_unlabeled, num_iterations=10, initial_threshold = .7):
        threshold = initial_threshold

        X_train = X_source.copy()
        y_prob_train = y_prob_source.copy()
        X_remaining = X_unlabeled.copy()

        self.model = DecisionTree(max_depth=5, min_samples_split=2)

        for iteration in range(num_iterations):
            if len(X_remaining) == 0:
                print(f"Done with self-training after {iteration + 1} iterations.")
                break

            # train model on current X_train
            self.model.fit(X_train, y_prob_train)

            # assign soft labels to unlabeled data
            probas = self.model.predict_proba(X_remaining)
            
            # find confident predictions
            max_probas = np.max(probas, axis=1)
            confident_idx = max_probas > threshold

            if not np.any(confident_idx):
                threshold = max(0.5, threshold - .02)
                print(f"Skipping iteration {iteration}, adjusting to a threshold of {threshold:.2f}")
                if threshold <= .6:
                    print(f"We were unable to give confident predictions on {len(X_remaining)} instances :(")
                    break

                continue

                
            
            print(f"Iteration {iteration} added {np.sum(confident_idx)} predictions.")
            # add samples to this new set of instances
            # that are above the confidence threshhold.
            # ( numpy boolean array indexing )
            new_X = X_remaining[confident_idx]
            # assign labels to this new data
            new_y_prob = probas[confident_idx]

            X_train = np.vstack((X_train, new_X))
            y_prob_train = np.vstack((y_prob_train, new_y_prob))

            # keep the samples that we are not yet confident about
            X_remaining = X_remaining[~confident_idx]
            

def main():
    classifier = SyntheticClassifier()
    X_source, y_prob_source, X_unlabeled, _, X_test, y_prob_test, y_test = classifier.load_and_prepare_data()

    # baseline model accuracy:
    baseline_model = DecisionTree(max_depth=5, min_samples_split=2)
    baseline_model.fit(X_source, y_prob_source)
    baseline_pred = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"Baseline model accuracy: {baseline_accuracy * 100}%")



    classifier.gradual_self_train(X_source, y_prob_source, X_unlabeled)

    y_pred = classifier.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Self-Trained model accuracy: {accuracy * 100}%")




if __name__ == "__main__":
    main()