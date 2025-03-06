# Model Optimization

This repository contains resources and code examples for machine learning model optimization techniques to enhance model efficiency, speed, and deployment capabilities, especially on systems without GPUs.

## Code Examples

This repository includes:

1. **MNIST Training:** A minimal CNN model for MNIST digit recognition that runs efficiently on CPU
2. **Model Optimization:** Implementation of multiple optimization techniques demonstrated on the MNIST model

To run the examples:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the original model
python mnist_training.py

# Apply optimization techniques
python model_optimization.py
```

## Optimization Techniques

### Model Quantization

Quantization is a technique used to reduce the size and memory footprint of neural network models. It involves converting the weights and activations of a neural network from high-precision floating-point numbers to lower-precision formats, such as 16-bit or 8-bit integers.

**Post-Training Quantization**

After training a deep learning model with high precision, you modify it to use less precise calculations. It's like taking a fully trained chef and asking them to cook using a smaller set of kitchen tools. Model accuracy and performance degrades in this case.

**Quantization-Aware Training**

During the training process, the model is aware that it will operate with less precise calculations, and it learns to handle this from the beginning. It's like training a chef from the start to be efficient with a specific set of kitchen tools. Model accuracy and performance decreases but performs better than Post-Training Quantization.

### Model Pruning

Model pruning refers to the technique for improving the efficiency and complexity of machine learning models by removing unnecessary parameters. 

**Weight Pruning** *(Unstructured Pruning)*

The idea behind weight pruning involves removing individual weights or connections within a neural network that are not contributing significantly to the model's performance.

**Neuron Pruning** *(Structured Pruning)*

Neuron pruning involves removing entire neurons from a neural network that are not contributing significantly to the model's performance. This can be done at the layer level or the channel level in convolutional neural networks.

### Knowledge Distillation

Knowledge distillation transfers knowledge from a large model (teacher) to a smaller model (student). Types include:

- Response-Based Knowledge Distillation (Logit Distillation)
- Feature-Based Knowledge Distillation (Intermediate Representation Transfer)
- Relation-Based Knowledge Distillation
- Self-Knowledge Distillation
- Cross-Model Knowledge Distillation
- Online Knowledge Distillation
- Multi-Teacher Knowledge Distillation

## Challenges

Every technique has its own challenges and trade-offs. Some of the common challenges include:

- Loss of Precision and Accuracy
- Impact on Performance
- Implementation complexity - Layer-wise, adaptive, rounding, etc.

## Resources

- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [PyTorch Tutorials on Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Optimization Guide](https://onnxruntime.ai/docs/performance/model-optimizations.html)
- [Neural Network Distiller by Intel](https://github.com/IntelLabs/distiller)
- [Awesome Model Compression Papers](https://github.com/cedrickchee/awesome-ml-model-compression)