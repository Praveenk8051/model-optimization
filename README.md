# Model Optimization

Optimising ML Models for efficient deployment, especially on systems without GPUs involves several key techniques. This notebook will cover the following:

1. Model Quantization
2. Model Pruning
3. Knowledge Distillation

# Challenges:

Every technique has its own challenges and trade-offs. Some of the common challenges include:

- Loss of Precision and Accuracy
- Impact on Performance
- Challenging techniques-Layer-wise, adaptive, rouding, etc.

## Model Quantization

Quantization is a technique used to reduce the size and memory footprint of neural network models. It involves converting the weights and activations of a neural network from high-precision floating-point numbers to lower-precision formats, such as 16-bit or 8-bit integers.

**Post-Training Quantization**

After training a deep learning model with high precision, you modify it to use less precise calculations. It's like taking a fully trained chef and asking them to cook using a smaller set of kitchen tools. Model accuracy and performance degrades in this case.

**Quantization-Aware Training**

During the training process, the model is aware that it will operate with less precise calculations, and it learns to handle this from the beginning. It's like training a chef from the start to be efficient with a specific set of kitchen tools. Model accuracy and performance decreases but performs better than Post-Training Quantization.

## Model Pruning

Model pruning refers to the technique for improving the efficiency and complexity of machine learning models by removing unnecessary parameters. 


**Weight Pruning** *(Unstructured Pruning)*

The idea behind weight pruning involves removing individual weights or connections within a neural network that are not contributing significantly to the model’s performance.

**Neuron Pruning** *(Structured Pruning)*

Neuron pruning involves removing entire neurons from a neural network that are not contributing significantly to the model’s performance. This can be done at the layer level or the channel level in convolutional neural networks.

Layer Pruning

## Knowledge Distillation

Response-Based Knowledge Distillation (Logit Distillation)

Feature-Based Knowledge Distillation (Intermediate Representation Transfer)

Relation-Based Knowledge Distillation

Self-Knowledge Distillation

Cross-Model Knowledge Distillation

Online Knowledge Distillation

Multi-Teacher Knowledge Distillation