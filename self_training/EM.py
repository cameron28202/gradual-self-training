import os
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
import re

class EM_Processor:
    def __init__(self, data_dir, image_size=(64, 64), n_components=100):
        self.data_dir = data_dir
        self.image_size = image_size
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def preprocess_image(self, file_path):
        '''
        1. Converts image to greyscale
        2. Resizes image
        3. Flattens image, returns the 1d array
        '''
        with Image.open(file_path) as img:
            img = img.convert('L')
            img = img.resize(self.image_size)
            return np.array(img).flatten()

    def process_images(self):
        '''
        Returns 3 arrays:
        1. X: array of preprocessed images
        2. y: labels of images (0 for F, 1 for M)
        3. years: years extracted from the name of the image files
        '''
        X, y, years = [], [], []
        for gender in ['F', 'M']:
            gender_dir = os.path.join(self.data_dir, gender)
            #for file_name in os.listdir(gender_dir):
            file_name = os.listdir(gender_dir)[0]
            if file_name.endswith('png'):
                year = int(re.search(r'(\d{4})', file_name).group(1))
                features = self.preprocess_image(os.path.join(gender_dir, file_name))
                X.append(features)
                y.append(0 if gender == 'F' else 1)
                years.append(year)

        X = np.array(X)
        y = np.array(y)
        years = np.array(years)

        X_pca = self.pca.fit_transform(X)

        return X_pca, y, years
    
    def split_data(self, X, y, years, labeled_years=(1905, 1935), unlabeled_years=(1935, 1969), test_years=(1969, 2013)):
        pass

def main():
    data_dir = 'data/classes'
    preprocessor = EM_Processor(data_dir, n_components = 1)
    X, y, years = preprocessor.process_images()

    print(f"{X[1]} : first image")
    print(f"{y[1]} : first image label")
    print(years[1])


    print("Processed images.")
    
if __name__ == "__main__":
    main()