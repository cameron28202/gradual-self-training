import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score



class ImpreciseClassifier:

    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
    
    def create_domains(self, n_domains=5, n_samples=2500):
        """
        Create a series of sequential, non-overlapping domains without rotation.
        
        :param n_domains: Number of domains to create (including source and target)
        :param n_samples: Number of samples to use from MNIST
        :return: List of tuples (X, y) for each domain
        """
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target

        # Convert to numpy arrays and ensure correct data type
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Subsample if necessary
        if n_samples and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]

        # Normalize pixel values
        X = X / 255.0

        # Create binary labels (0 for even, 1 for odd)
        y_binary = (y % 2).astype(int)

        # Calculate samples per domain
        samples_per_domain = len(X) // n_domains

        # Create domains
        domains = []
        for i in range(n_domains):
            start_idx = i * samples_per_domain
            end_idx = start_idx + samples_per_domain if i < n_domains - 1 else len(X)
            domains.append((X[start_idx:end_idx], y_binary[start_idx:end_idx]))

        return domains

    def create_rotated_domains(self, n_domains=5, max_rotation=60, n_samples=1000):
        """
        Create a series of domains with gradually increasing rotation.
        
        :param n_domains: Number of domains to create (including source and target)
        :param max_rotation: Maximum rotation angle for the target domain
        :param n_samples: Number of samples to use from MNIST
        :return: List of tuples (X, y) for each domain
        """
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target

        # Convert to numpy arrays and ensure correct data type
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

        # Normalize pixel values
        X = X / 255.0

        # Create binary labels (0 for even, 1 for odd)
        y_binary = (y % 2).astype(int)

        # Split into train (source) and test (target) sets
        X_source, X_target, y_source, y_target = train_test_split(X, y_binary, test_size=0.2, random_state=42)

        # Calculate rotation angles for each domain
        rotation_angles = np.linspace(0, max_rotation, n_domains)

        # Create domains
        domains = []
        for i, angle in enumerate(rotation_angles):
            if i == 0:
                # Source domain (no rotation)
                domains.append((X_source, y_source))
            elif i == n_domains - 1:
                # Target domain (maximum rotation)
                domains.append((rotate_images(X_target, angle), y_target))
            else:
                # Intermediate domains
                domains.append((rotate_images(X_source, angle), y_source))

        return domains
    
    def train_imprecise(self, domains, imprecision_percent=5, max_iterations=10, convergence_threshold=1e-4):
        # Train on source domain
        X_source, y_source = domains[0]
        self.model.fit(X_source, y_source)

        # Unpack target domain
        X_target, y_target = domains[-1]

        # Make initial prediction on target domain
        baseline_preds = self.model.predict(X_target)
        baseline_accuracy = accuracy_score(baseline_preds, y_target)

        print(f"Baseline model accuracy: {baseline_accuracy}")
        

        for domain_index, (X, _) in enumerate(domains[1:-1], start=1):
            print(f"Processing domain {domain_index}")
            
            # Generate pseudo-labels for this domain
            pseudo_labels = self.model.predict(X)
            
            # Add imprecision to the intermediate domain's pixel values
            X_imprecise = np.array([[add_imprecision(pixel, imprecision_percent) for pixel in instance] for instance in X])
            
            # Select random precise pixel values from the imprecise range
            X_precise = np.array([[np.random.uniform(pixel[0], pixel[1]) for pixel in instance] for instance in X_imprecise])
            
            for iteration in range(max_iterations):
                # Train model on current precise values and pseudo-labels
                self.model.fit(X_precise, pseudo_labels)

                for i, imprecise_instance in enumerate(X_imprecise):
                    for j, pixel_range in enumerate(imprecise_instance):
                        test_values = [pixel_range[0], np.mean(pixel_range), pixel_range[1]]
                        instance_copy = X_precise[i].copy()
                        probas = []
                        for val in test_values:
                            instance_copy[j] = val
                            probas.append(self.model.predict_proba(instance_copy.reshape(1, -1))[0])

                        best_value = test_values[np.argmax([np.max(p) for p in probas])]
                        X_precise[i, j] = best_value

                        # Update pseudo-label based on the new instance
                        pseudo_labels[i] = np.argmax(self.model.predict_proba(X_precise[i].reshape(1, -1))[0])
                
                # Evaluate on target domain (in practice, you might do this less frequently)
                current_pred = self.model.predict(X_target)
                current_accuracy = accuracy_score(current_pred, y_target)
                print(f"Iteration {iteration} accuracy: {current_accuracy:.4f}")

            print(f"Finished processing domain {domain_index}")
            print(f"Final accuracy for domain {domain_index}: {current_accuracy:.4f}")
            print("-----------------------------")
                            

def add_imprecision(pixel_value, imprecision_percent=5):
    lower_bound = max(0, pixel_value - (pixel_value * imprecision_percent / 100))
    upper_bound = min(1, pixel_value + (pixel_value * imprecision_percent / 100))
    return [lower_bound, upper_bound]
        

def rotate_images(images, angle):
    """
    Rotate a batch of MNIST images by a given angle.
    
    :param images: numpy array of shape (n_samples, 784) containing flattened MNIST images
    :param angle: rotation angle in degrees
    :return: numpy array of same shape as input, containing rotated images
    """
    # Reshape images to (n_samples, 28, 28)
    n_samples = images.shape[0]
    images_reshaped = images.reshape(n_samples, 28, 28)
    
    rotated_images = np.zeros_like(images_reshaped)
    
    for i in range(n_samples):
        # Get the image
        img = images_reshaped[i]
        
        # Get the image center
        center = (14, 14)
        
        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        rotated_images[i] = rotated
    
    # Reshape back to (n_samples, 784)
    return rotated_images.reshape(n_samples, 784)



def main():
    classifier = ImpreciseClassifier()
    domains = classifier.create_rotated_domains()

    classifier.train_imprecise(domains)
    

if __name__ == "__main__":
    main()

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score



class ImpreciseClassifier:

    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
    
    def create_domains(self, n_domains=5, n_samples=2500):
        """
        Create a series of sequential, non-overlapping domains without rotation.
        
        :param n_domains: Number of domains to create (including source and target)
        :param n_samples: Number of samples to use from MNIST
        :return: List of tuples (X, y) for each domain
        """
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target

        # Convert to numpy arrays and ensure correct data type
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Subsample if necessary
        if n_samples and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]

        # Normalize pixel values
        X = X / 255.0

        # Create binary labels (0 for even, 1 for odd)
        y_binary = (y % 2).astype(int)

        # Calculate samples per domain
        samples_per_domain = len(X) // n_domains

        # Create domains
        domains = []
        for i in range(n_domains):
            start_idx = i * samples_per_domain
            end_idx = start_idx + samples_per_domain if i < n_domains - 1 else len(X)
            domains.append((X[start_idx:end_idx], y_binary[start_idx:end_idx]))

        return domains

    def create_rotated_domains(self, n_domains=3, max_rotation=60, n_samples=10000):
        """
        Create a series of domains with gradually increasing rotation.
        
        :param n_domains: Number of domains to create (including source and target)
        :param max_rotation: Maximum rotation angle for the target domain
        :param n_samples: Number of samples to use from MNIST
        :return: List of tuples (X, y) for each domain
        """
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target

        # Convert to numpy arrays and ensure correct data type
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

        # Normalize pixel values
        X = X / 255.0

        # Create binary labels (0 for even, 1 for odd)
        y_binary = (y % 2).astype(int)

        # Split into train (source) and test (target) sets
        X_source, X_target, y_source, y_target = train_test_split(X, y_binary, test_size=0.2, random_state=42)

        # Calculate rotation angles for each domain
        rotation_angles = np.linspace(0, max_rotation, n_domains)

        # Create domains
        domains = []
        for i, angle in enumerate(rotation_angles):
            if i == 0:
                # Source domain (no rotation)
                domains.append((X_source, y_source))
            elif i == n_domains - 1:
                # Target domain (maximum rotation)
                domains.append((rotate_images(X_target, angle), y_target))
            else:
                # Intermediate domains
                domains.append((rotate_images(X_source, angle), y_source))

        return domains
    
    def train_imprecise(self, domains, imprecision_percent=5, max_iterations=10, convergence_threshold=1e-4):

        # TODO : 
        # Implement Imprecise Labels
        # See if prediction on an imprecise target dataset is more accurate than a precise one

        # train initial model on precise labeled source data
        X_source, y_source = domains[0]
        self.model.fit(X_source, y_source)

        # unpack target domain
        X_target, y_target = domains[-1]

        baseline_pred = self.model.predict(X_target)
        baseline_accuracy = accuracy_score(baseline_pred, y_target)
        print(f"Initial model accuracy: {baseline_accuracy:.2f}")

        for domain_index, (X, y) in enumerate(domains[1:-1], start=1):
            print(f"Processing domain {domain_index}")
            
            # add imprecision to the intermediate domain's pixel values
            X_imprecise = np.array([[add_imprecision(pixel, imprecision_percent) for pixel in instance] for instance in X])

            # select random precise pixel values from the imprecise range
            X_precise = np.array([[np.random.uniform(pixel[0], pixel[1]) for pixel in instance] for instance in X_imprecise])
            
            prev_loss = float('inf')
            for iteration in range(max_iterations):
                # train model on current precise 
                self.model.fit(X_precise, y)

                for i, imprecise_instance in enumerate(X_imprecise):
                    best_instance = []
                    for j, pixel_range in enumerate(imprecise_instance):
                        # test values are lower bound, midpoint, and upper bound of the imprecise range
                        test_values = [pixel_range[0], np.mean(pixel_range), pixel_range[1]]

                        # find the probas for each of these tests
                        instance_copy = X_precise[i].copy()
                        probas = []
                        for val in test_values:
                            instance_copy[j] = val
                            probas.append(self.model.predict_proba(instance_copy.reshape(1, -1))[0, y[i]])
                        
                        # find which of the 3 had the best proba
                        best_value = test_values[np.argmax(probas)]
                        best_instance.append(best_value)
                    X_precise[i] = best_instance
                    
                # check for convergence
                current_loss = -np.mean(np.log(self.model.predict_proba(X_precise)[np.arange(len(y)), y]))
                if abs(prev_loss - current_loss) < convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break
                prev_loss = current_loss
                print(f"Current loss: {current_loss}")
                current_pred = self.model.predict(X_target)
                current_accuracy = accuracy_score(current_pred, y_target)
                print(f"Iteration {iteration} accuracy: {current_accuracy:.4f}")

            print(f"Finished processing domain {domain_index}")
            print(f"Final accuracy for domain {domain_index}: {current_accuracy:.4f}")
            print("-----------------------------")
                            

def add_imprecision(pixel_value, imprecision_percent=5):
    lower_bound = max(0, pixel_value - (pixel_value * imprecision_percent / 100))
    upper_bound = min(1, pixel_value + (pixel_value * imprecision_percent / 100))
    return [lower_bound, upper_bound]
        

def rotate_images(images, angle):
    """
    Rotate a batch of MNIST images by a given angle.
    
    :param images: numpy array of shape (n_samples, 784) containing flattened MNIST images
    :param angle: rotation angle in degrees
    :return: numpy array of same shape as input, containing rotated images
    """
    # Reshape images to (n_samples, 28, 28)
    n_samples = images.shape[0]
    images_reshaped = images.reshape(n_samples, 28, 28)
    
    rotated_images = np.zeros_like(images_reshaped)
    
    for i in range(n_samples):
        # Get the image
        img = images_reshaped[i]
        
        # Get the image center
        center = (14, 14)
        
        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        rotated_images[i] = rotated
    
    # Reshape back to (n_samples, 784)
    return rotated_images.reshape(n_samples, 784)



def main():
    classifier = ImpreciseClassifier()
    domains = classifier.create_domains()

    classifier.train_imprecise(domains)
    

if __name__ == "__main__":
    main()
