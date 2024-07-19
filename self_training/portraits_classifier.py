import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from collections import defaultdict

import os
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
import re

class DecisionTree():
    pass

class PortraitsClassifier():
    def __init__(self, data_dir, image_size=(64, 64), n_components=100):
        self.data_dir = data_dir
        self.image_size = image_size
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        base_cart = DecisionTreeClassifier(max_depth=5)
        self.baseline_model = CalibratedClassifierCV(base_cart, method='sigmoid', cv=5)
        self.model = CalibratedClassifierCV(base_cart, method='sigmoid', cv=5)
        self.scaler = StandardScaler()

    def preprocess_image(self, file_path):
        with Image.open(file_path) as img:
            img = img.convert('L')
            img = img.resize(self.image_size)
            return np.array(img).flatten()
    
    def prepare_image(self, images_per_year = 20):
        X, y, years = [], [], []
        year_counter = defaultdict(int)
        for gender in ['F', 'M']:
            year_counter.clear()
            gender_dir = os.path.join(self.data_dir, gender)
            for file_name in os.listdir(gender_dir):
                if file_name.endswith('png'):
                    year = int(re.search(r'(\d{4})', file_name).group(1))
                    if year_counter[year] <= images_per_year:
                        year_counter[year] += 1                  
                        features = self.preprocess_image(os.path.join(gender_dir, file_name))
                        X.append(features)
                        y.append(0 if gender == 'F' else 1)
                        years.append(year)
                        #print(f"Processing {gender} : {file_name}, Year: {year}")
            

        X = np.array(X)
        y = np.array(y)
        years = np.array(years)

        X_pca = self.pca.fit_transform(X)

        random_indices = np.random.permutation(len(X_pca))
        X_pca = X_pca[random_indices]
        y = y[random_indices]
        years = years[random_indices]

        return X_pca, y, years

    def split_data(self, X, y, years, test_size=.2, unlabeled_size=.6, random_state=42):
        
        '''
            test size: determines proportion of data held out for final testing
            unlabeled_size: determines proportion of total data that will be unlabeled
        '''


        # by default, 20% of data is allocated into test set
        # X_test, y_test and years_test are new test sets
        # X_temp, y_temp and years_temp are the remaining 80% data
        X_temp, X_test, y_temp, y_test, years_temp, years_test = train_test_split(
            X, y, years, test_size=test_size, random_state=random_state, stratify=y
        )

        labeled_size = 1 - (test_size + unlabeled_size)
        labeled_ratio = labeled_size / (1 - test_size)
        
        # by default, we are taking the rest of the 80% daya and splitting it
        # into 20% source (labeled) and 60% unlabeled.
        # X_source, y_source and years_source are now labeled datasets
        # X_unlabeled, y_unlabeled and years_unlabeled are now unlabeled datasets
        X_labeled, X_unlabeled, y_labeled, y_unlabeled, years_labeled, years_unlabeled = train_test_split(
            X_temp, y_temp, years_temp, test_size=1-labeled_ratio, random_state=random_state, stratify=y_temp
        )

        return (X_labeled, y_labeled, years_labeled), (X_unlabeled, y_unlabeled, years_unlabeled), (X_test, y_test, years_test)

    def baseline_train(self, source_data):
        X_train = source_data[0]
        y_train = source_data[1]
        self.baseline_model.fit(X_train, y_train)

    def self_train(self, source_data, unlabeled_data, n_iterations = 10, initial_threshold = .7):


        '''
            Expected paramaters:
            source_data: (X_source, y_source, years_source)
            unlabeled_data: (X_unlabeled, y_unlabeled, years_unlabeled)
        '''
        threshold = initial_threshold
        X_train = source_data[0]
        y_train = source_data[1]
        years_train = source_data[2]

        X_unlabeled = unlabeled_data[0]
        years_unlabeled = unlabeled_data[2]

        num_confident = []

        for iteration in range(n_iterations):

            if len(X_unlabeled) == 0:
                print(f"Done w/ self-training after iteration {iteration}")
                break

            # train model on current labeled data
            self.model.fit(X_train, y_train)

            # create soft labels for current unlabeled predictions
            probas = self.model.predict_proba(X_unlabeled)

            # find the best prediction the model had
            max_probas = np.max(probas, axis=1)

            # store the indices with confident predictions
            confident_idx = max_probas > threshold

            num_confident = np.sum(confident_idx)

            if not np.any(confident_idx):
                print(f"No confident predictions after {iteration + 1} iterations. Stopping self-training.")
                break

            print(f"Iteration {iteration + 1} added {num_confident} confident predictions.")

            # instances that we're confident enough to assign labels
            new_X = X_unlabeled[confident_idx]

            # assign labels to these new instances
            new_y = self.model.predict(new_X)
            new_years = years_unlabeled[confident_idx]

            # add the new data to the labeled dataset
            X_train = np.vstack((X_train, new_X))
            y_train = np.hstack((y_train, new_y))
            years_train = np.hstack((years_train, new_years))

            # keep the instances we aren't yet sure about in the unlabeled dataset
            X_unlabeled = X_unlabeled[~confident_idx]
            years_unlabeled = years_unlabeled[~confident_idx]
            threshhold = min(threshhold + .05, .95)


def main():
    classifier = PortraitsClassifier("data\classes")
    X, y, years = classifier.prepare_image()
    source_data, unlabeled_data, test_data = classifier.split_data(X, y, years)

    # unpack test data
    X_test = test_data[0]
    y_test = test_data[1]
    year_test = test_data[2]

    # train baseline model, analyze accuracy
    classifier.baseline_train(source_data)

    baseline_pred = classifier.baseline_model.predict(X_test)
    accuracy = accuracy_score(y_test, baseline_pred)
    print(f"Baseline Model's accuracy: {accuracy * 100}%.")

    # train self-trained model, analyze accuracy
    classifier.self_train(source_data, unlabeled_data)

    pred = classifier.model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Self-Trained Model's accuracy: {accuracy * 100}%.")

    # pred = classifier.model.predict(X_test[:20])
    # for x in range(20):
    #     print(f"Predicted that image {x+1} was {pred[x]}.")
    #     print("Correct" if pred[x] == y_test[x] else "Incorrect")
        

    

if __name__ == "__main__":
    main()
