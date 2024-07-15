import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

class SyntheticClassifier:
    def __init__(self):
        base_cart = DecisionTreeClassifier(max_depth=5)
        self.baseline_model = CalibratedClassifierCV(base_cart, method='sigmoid', cv=5)
        self.model = CalibratedClassifierCV(base_cart, method='sigmoid', cv=5)
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, n_samples=100000):

        # binary classification
        n_features = 2
        # class 0 centered around -1, -1
        class_0 = np.random.randn(n_samples // 2, n_features) + [-1, -1]
        # class 1 centered around 1, 1
        class_1 = np.random.randn(n_samples // 2, n_features) + [1, 1]
        # stack into single 2d array
        X = np.vstack((class_0, class_1))
        # stack into single labels array corresponding to x arr
        y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
        # add random noise
        X += np.random.randn(n_samples, n_features) * .5
        # shuffle
        random_indices = np.random.permutation(n_samples)
        X = X[random_indices]
        y = y[random_indices]

        return X, y
    
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


    def baseline_train(self, X_train, Y_train):
        self.baseline_model.fit(X_train, Y_train)

    def self_train(self, X_labeled, y_labeled, X_unlabeled, n_iterations=10, initial_threshold=.8):

        threshold = initial_threshold
        # copy initial labeled dataset then the remaining instances
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_remaining = X_unlabeled.copy()

        # number of labeled counts
        # use an array to track how the number of 
        # labeled samples changes at each iter
        labeled_counts = [len(X_labeled)]

        for iteration in range(n_iterations):
            if len(X_remaining) == 0:
                print(f"Done w/ self-training after iteration {iteration}")
                break
            # train model on current labeled data
            self.model.fit(X_train, y_train)

            # assign soft labels to unlabeled data 
            probas = self.model.predict_proba(X_remaining)
            
            max_probas = np.max(probas, axis=1)
            confident_idx = max_probas > threshold

            if not np.any(confident_idx):
                print(f"No confident predictions in iteration {iteration + 1}. Stopping.")
                break
            
            # add samples to this new set of instances
            # that are above the confidence threshhold.
            # ( numpy boolean array indexing )
            new_X = X_remaining[confident_idx]
            # assign labels to this new data
            new_y = self.model.predict(new_X)

            X_train = np.vstack((X_train, new_X))
            y_train = np.hstack((y_train, new_y))

            # keep the samples that we are not yet confident about
            X_remaining = X_remaining[~confident_idx]

            labeled_counts.append(len(X_train))
            threshold = min(threshold + 0.02, 0.95)


    
def main():
    classifier = SyntheticClassifier()

    X_source, y_source, X_unlabeled, _, X_test, y_test = classifier.load_and_prepare_data()

    # train baseline model
    classifier.baseline_train(X_source, y_source)

    baseline_pred = classifier.baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"Baseline model's accuracy: {baseline_accuracy * 100}%")

    # train self-training model
    classifier.self_train(X_source, y_source, X_unlabeled)

    pred = classifier.model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print(f"Self-Training model's accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    main()
    
