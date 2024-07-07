from classifier import CoverTypeClassifier

class Driver:
    def __init__(self):
        self.classifier = CoverTypeClassifier()
    
    def run(self):
        # Load and prepare data
        X_source_scaled, X_intermediate_scaled, X_target_scaled, y_source, y_intermediate, y_target = self.classifier.load_and_prepare_data()
        # Train the model
        print("Training the model...")
        final_target_accuracy = self.classifier.gradual_self_train(
            X_source_scaled, y_source, 
            X_intermediate_scaled, y_intermediate, 
            X_target_scaled, y_target
        )
        print(f"Final accuracy on target domain: {final_target_accuracy}")

        baseline_target_accuracy = self.classifier.baseline_train(X_source_scaled, y_source, X_target_scaled, y_target)

        print(f"Baseline accuracy on target domain: {baseline_target_accuracy}")
        
if __name__ == "__main__":
    driver = Driver()
    driver.run()