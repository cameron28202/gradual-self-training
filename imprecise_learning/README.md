# Imprecise Learning for MNIST Binary Classification

## Project Overview

This project implements an imprecise learning algorithm for binary classification (even vs. odd) of MNIST handwritten digits. The implementation is inspired by the paper "Self-Domain Adaptation Through Imprecisiation" and explores the potential of learning from imprecise data to improve model robustness and generalization.

## Current Implementation

### Key Features

1. **Data Preparation**:
   - MNIST dataset is loaded and preprocessed.
   - Binary labels are created (0 for even, 1 for odd digits).
   - A gradual domain shift is implemented by shifting each domain by max_rotation / n_domains degrees.
   - Source domain is not rotated 
   - Target domain is fully rotated

2. **Imprecise Data Representation**:
   - Pixel values are represented as ranges rather than precise values.
   - Imprecision is added using a percentage-based approach.

3. **Alternating Optimization**:
   - The algorithm alternates between updating the model parameters, then selecting optimal precise instances from imprecise ranges.

4. **Logistic Regression Model**:
   - A simple logistic regression model is used as the base classifier.

### Imprecise Learning Process

1. Initial model training on precise source data.
2. For each subsequent domain:
   - Add imprecision to pixel values.
   - Randomly select initial precise values from the imprecise ranges.
   - Iteratively:
     a. Train model on current precise instances.
     b. Select best precise instances by testing lower bound, midpoint, and upper bound of each pixel's imprecise range.
   - Continue until convergence or max iterations reached.

## Preliminary Findings

- The model sometimes shows improvement in accuracy after training on imprecise data.
- However, accuracy often decreases after each iteration of adding imprecision. I want to explore why this is happening, maybe the model is overfitting to the imprecise data, making it less accurate on precise target domain?
- Convergence is typically achieved within a few iterations for each domain.
- Performance varies across domains, suggesting potential for further optimization.

### Sample Output for 5 domains, 5000 samples

```
Initial model accuracy: 86.25%
Processing domain 1
Current loss: 0.06793042622416212
Iteration 0 accuracy: 0.8585
Current loss: 0.03445668401801197
Iteration 1 accuracy: 0.8305
Current loss: 0.0346398298320042
Iteration 2 accuracy: 0.8315
Converged after 4 iterations
Finished processing domain 1
Final accuracy for domain 1: 0.8315
-----------------------------
Processing domain 2
Current loss: 0.05505136333348069
Iteration 0 accuracy: 0.8655
Current loss: 0.029618274337404108
Iteration 1 accuracy: 0.8450
Current loss: 0.029259480523583264
Iteration 2 accuracy: 0.8440
Converged after 4 iterations
Finished processing domain 2
Final accuracy for domain 2: 0.8440
-----------------------------
Processing domain 3
Current loss: 0.0553793459592318
Iteration 0 accuracy: 0.8445
Current loss: 0.029782983844801558
Iteration 1 accuracy: 0.8170
Converged after 3 iterations
Finished processing domain 3
Final accuracy for domain 3: 0.8170
-----------------------------
Processing domain 4
Current loss: 0.06359807445840113
Iteration 0 accuracy: 0.9280
Current loss: 0.0325157443099753
Iteration 1 accuracy: 0.9020
Current loss: 0.03207662462516234
Iteration 2 accuracy: 0.9045
Converged after 4 iterations
Finished processing domain 4
Final accuracy for domain 4: 0.9045
-----------------------------
Final model accuracy: 90.45%.
```

## Future Goals

1. **Optimization and Refinement**:
   - Improve the process of finding optimal precise values within imprecise ranges.
   - Explore methods to consistently improve accuracy across iterations.

2. **Imprecise Labels**:
   - Extend the implementation to include imprecision in labels, not just features.

3. **Performance Analysis**:
   - Conduct a more comprehensive analysis of the algorithm's performance under various conditions.

## Current Challenges

- High computational cost of finding optimal precise values within imprecise ranges.
- Inconsistent improvement in accuracy across iterations and domains.
