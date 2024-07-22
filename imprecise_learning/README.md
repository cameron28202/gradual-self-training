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

### Sample Output for 7 domains, 10000 samples, max rotation of 15 degrees

```
Baseline model accuracy: 0.8207
Processing domain 1
Iteration 0 accuracy: 0.8270
Iteration 1 accuracy: 0.8245
Iteration 2 accuracy: 0.8273
Iteration 3 accuracy: 0.8273
Converged after 4 iterations
Finished processing domain 1
Final accuracy for domain 1: 0.8273
-----------------------------
Processing domain 2
Iteration 0 accuracy: 0.8307
Iteration 1 accuracy: 0.8317
Iteration 2 accuracy: 0.8347
Iteration 3 accuracy: 0.8333
Iteration 4 accuracy: 0.8310
Iteration 5 accuracy: 0.8325
Iteration 6 accuracy: 0.8323
Iteration 7 accuracy: 0.8347
Iteration 8 accuracy: 0.8337
Iteration 9 accuracy: 0.8340
Finished processing domain 2
Final accuracy for domain 2: 0.8340
-----------------------------
...
-----------------------------
Processing domain 5
Iteration 0 accuracy: 0.8390
Iteration 1 accuracy: 0.8360
Iteration 2 accuracy: 0.8370
Iteration 3 accuracy: 0.8365
Iteration 4 accuracy: 0.8370
Iteration 5 accuracy: 0.8395
Iteration 6 accuracy: 0.8373
Iteration 7 accuracy: 0.8403
Iteration 8 accuracy: 0.8385
Iteration 9 accuracy: 0.8367
Finished processing domain 5
Final accuracy for domain 5: 0.8367
-----------------------------
Final model accuracy: 0.8367
Improvement: 0.0160
```

## Future Goals

1. **Optimization and Refinement**:
   - Improve the process of finding optimal precise values within imprecise ranges.
   - Explore methods to consistently improve accuracy across iterations.

2. **Imprecise Labels**:
   - Extend the implementation to include imprecision in labels, not just features.

3. **Performance Analysis**:
   - Conduct a more comprehensive analysis of the algorithm's performance under various conditions.
