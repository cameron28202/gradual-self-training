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

    def create_rotated_domains(self, n_domains=7, max_rotation=15, n_samples=20000):
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
    
    def train_imprecise(self, domains, imprecision_percent=5, max_iterations=10, convergence_threshold=1e-4, pixel_batch_size=100):
        # Train on source domain
        X_source, y_source = domains[0]
        self.model.fit(X_source, y_source)

        # Unpack target domain
        X_target, y_target = domains[-1]

        # Make initial prediction on target domain
        baseline_preds = self.model.predict(X_target)
        baseline_accuracy = accuracy_score(baseline_preds, y_target)
        print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

        for domain_index, (X, _) in enumerate(domains[1:-1], start=1):
            print(f"Processing domain {domain_index}")
            
            # Generate pseudo-labels for this domain
            pseudo_labels = self.model.predict(X)
            
            # Add imprecision to the intermediate domain's pixel values
            X_imprecise = add_imprecision(X, imprecision_percent)
            
            # Select random precise pixel values from the imprecise range
            X_precise = np.random.uniform(X_imprecise[:,:,0], X_imprecise[:,:,1])
            
            
            prev_accuracy = 0
            for iteration in range(max_iterations):
                # Train model on current precise values and pseudo-labels
                self.model.fit(X_precise, pseudo_labels)

                # Process pixels in batches
                pixel_indices = np.random.permutation(784)
                for start in range(0, 784, pixel_batch_size):
                    end = min(start + pixel_batch_size, 784)
                    batch_indices = pixel_indices[start:end]
                    
                    # Create test values for all instances and pixels in the batch
                    test_values = np.stack([
                        X_imprecise[:, batch_indices, 0], # lower bound
                        (X_imprecise[:, batch_indices, 0] + X_imprecise[:, batch_indices, 1]) / 2, # middle pixel
                        X_imprecise[:, batch_indices, 1] # upper bound
                    ], axis=2)
                    
                    # Compute probabilities for all test values
                    probas = np.zeros((X_precise.shape[0], len(batch_indices), 3))
                    for i in range(3):
                        temp_X = X_precise.copy()
                        temp_X[:, batch_indices] = test_values[:, :, i]
                        probas[:, :, i] = np.max(self.model.predict_proba(temp_X), axis=1).reshape(-1, 1)
                    
                    # Select best values
                    best_indices = np.argmax(probas, axis=2)
                    X_precise[:, batch_indices] = test_values[np.arange(X_precise.shape[0])[:, None], np.arange(len(batch_indices)), best_indices]
                
                # Update pseudo-labels
                pseudo_labels = np.argmax(self.model.predict_proba(X_precise), axis=1)
                
                # Evaluate on target domain
                current_pred = self.model.predict(X_target)
                current_accuracy = accuracy_score(current_pred, y_target)
                print(f"Iteration {iteration} accuracy: {current_accuracy:.4f}")

                # Check for convergence
                if abs(current_accuracy - prev_accuracy) < convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break

                prev_accuracy = current_accuracy

            print(f"Finished processing domain {domain_index}")
            print(f"Final accuracy for domain {domain_index}: {current_accuracy:.4f}")
            print("-----------------------------")

        # Final evaluation
        final_preds = self.model.predict(X_target)
        final_accuracy = accuracy_score(final_preds, y_target)
        print(f"Final model accuracy: {final_accuracy:.4f}")
        print(f"Improvement: {final_accuracy - baseline_accuracy:.4f}")

def add_imprecision(pixel_values, imprecision_percent=5):
    lower_bound = np.maximum(0, pixel_values - (pixel_values * imprecision_percent / 100))
    upper_bound = np.minimum(1, pixel_values + (pixel_values * imprecision_percent / 100))
    return np.stack([lower_bound, upper_bound], axis=-1)
        

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

    #print(domains[0][0])
    classifier.train_imprecise(domains)
    

if __name__ == "__main__":
    main()

