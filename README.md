# Predictive Coding Neural Network

A custom implementation of predictive coding networks—biologically-inspired neural architectures that learn by minimizing prediction errors rather than through traditional backpropagation.

## Overview

This project implements predictive coding (PC) networks, a brain-inspired learning algorithm where networks learn to predict their inputs and adjust based on prediction errors. This approach more closely mirrors how biological brains are thought to process information compared to standard artificial neural networks.

Traditional neural networks use backpropagation, which is biologically implausible. Predictive coding offers:
- **Biological plausibility:** Matches neural circuitry better
- **Local learning:** Each layer learns based on local prediction errors
- **Hierarchical inference:** Models how the brain processes information across cortical layers
- **Unsupervised learning potential:** Can learn representations without labels

## Technical Stack

- **Language:** Python
- **Core Libraries:** NumPy
- **Framework:** Custom implementation (no TensorFlow/PyTorch dependency)
- **Visualization:** Matplotlib

## Key Features

### 1. Predictive Coding Learning Rule
```python
# Error calculation: difference between prediction and actual
error = actual - prediction

# Weight update based on prediction error
dw = learning_rate * error * input
```

### 2. Comparison with Standard ANNs
The notebook includes side-by-side comparison of:
- Predictive Coding Network (PC)
- Standard Artificial Neural Network (ANN) with backpropagation

### 3. Flexible Architecture
- Configurable number of layers
- Multiple activation functions (tanh, sigmoid, ReLU)
- Adjustable learning rates and iterations
- Customizable prediction error propagation

## Network Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
    ↓            ↓                 ↓
  Prediction   Prediction      Prediction
    ↑            ↑                 ↑
  Error        Error            Error
```

Each layer:
1. Makes a prediction about its input
2. Computes prediction error
3. Updates weights to minimize error
4. Passes refined prediction forward

## Performance

On sequence prediction tasks, the PC network achieves:
- **>90% accuracy** on unseen sequences
- **Comparable performance** to standard backpropagation
- **Better biological plausibility** with local learning rules

## Usage

```python
import numpy as np

# Define network parameters
params = {
    'type': 'tanh',           # Activation function
    'l_rate': 0.2,            # Learning rate
    'it_max': 100,            # Inference iterations
    'epochs': 500,            # Training epochs
    'beta': 0.2,              # Prediction error weight
    'neurons': [2, 5, 1]      # Network architecture
}

# Initialize weights
w, b = w_init(params)

# Train network
w_trained, b_trained = learn_pc(in_data, out_data, w, b, params)

# Test on new data
accuracy = test(test_in, test_out, w_trained, b_trained, params)
```

## Research Applications

This implementation is useful for:
- **Cognitive neuroscience:** Modeling cortical computation
- **AI research:** Exploring alternatives to backpropagation
- **Computational psychiatry:** Modeling prediction error signaling
- **Robotics:** Sensorimotor prediction and control
- **Unsupervised learning:** Learning representations without labels

## Theoretical Background

Predictive coding is grounded in:
- **Free energy principle** (Karl Friston)
- **Hierarchical predictive processing** in cortex
- **Bayesian brain hypothesis**
- **Error-driven learning** in neuroscience

### Key Papers
- Rao & Ballard (1999): Predictive coding in the visual cortex
- Friston (2005): A theory of cortical responses
- Millidge et al. (2021): Predictive coding approximates backprop along arbitrary computation graphs

## Comparison: PC vs Backpropagation

| Feature | Predictive Coding | Backpropagation |
|---------|------------------|-----------------|
| Biological plausibility | High | Low |
| Local learning | Yes | No |
| Online learning | Yes | Limited |
| Credit assignment | Prediction errors | Global error signal |
| Computational cost | Higher (iterative) | Lower (one pass) |

## Implementation Details

- **Inference phase:** Iterative prediction error minimization (typically 50-100 iterations)
- **Learning phase:** Weight updates based on converged predictions
- **Activation functions:** Supports tanh, sigmoid, ReLU
- **No automatic differentiation:** Uses explicit error propagation

## Experimental Results

Tested on:
- XOR problem: 100% accuracy
- Non-linear function approximation: >95% accuracy
- Sequential pattern recognition: >90% accuracy

## Future Directions

- Add convolutional layers for image processing
- Implement continuous-time predictive coding
- Add recurrent connections for temporal prediction
- Test on more complex datasets (MNIST, etc.)
- Compare with energy-based models

---

**Author:** Yasemin Gokcen  
**Collaborator/Advisor:** Dr. Jeffery Yoshimi
**Affiliation:** PhD Candidate, Cognitive & Information Sciences, UC Merced  
**Contact:** ygokcen@ucmerced.edu

## References

- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex
- Friston, K. (2005). A theory of cortical responses
- Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network
