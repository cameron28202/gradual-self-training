import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class SyntheticClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, n_samples=1000):

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
        X_source, X_temp, y_source, y_temp = train_test_split(X, y, test_size=.8, random_state=42)
        X_unlabeled, X_test, y_unlabeled, y_test = train_test_split(X_temp, y_temp, test_size=.5, random_state=42)

        # transform data
        X_source = self.scaler.fit_transform(X_source)
        X_unlabeled = self.scaler.transform(X_unlabeled)
        X_test = self.scaler.transform(X_test)

        return X_source, y_source, X_unlabeled, y_unlabeled, X_test, y_test


    def self_train(self, X_labeled, y_labeled, X_unlabeled, n_iterations=10, threshhold=.7):

        # copy initial labeled dataset then the remaining instances
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_remaining = X_unlabeled.copy()

        # number of labeled counts
        # use an array to track how the number of 
        # labeled samples changes at each iter
        labeled_counts = [len(X_labeled)]

        for _ in range(n_iterations):
            if len(X_remaining) == 0:
                print("Done w/ self-training")
                break
            # train model on current labeled data
            self.model.fit(X_train, y_train)

            # assign soft labels to unlabeled data 
            probas = self.model.predict_proba(X_remaining)
            
            max_probas = np.max(probas, axis=1)
            confident_idx = max_probas > threshhold
            
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
        
        return X_train, y_train, labeled_counts


    
def main():
    classifier = SyntheticClassifier()
    X_source, y_source, X_unlabeled, y_unlabeled, X_test, y_test = classifier.load_and_prepare_data()
    X_train, Y_train, labeled_counts = classifier.self_train(X_source, y_source, X_unlabeled)

    pred = classifier.model.predict(X_test)
    accuracy = 0
    for x in range(len(X_test)):
        accuracy += 1 if pred[x] == y_test[x] else 0

    print(f"Model's accuracy: {accuracy/len(X_test)}")


if __name__ == "__main__":
    main()
    
