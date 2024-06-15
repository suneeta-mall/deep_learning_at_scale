
# Chapter 2: Deep Learning

This chapter focuses on deep learning and includes two key exercises.

## Exercise 1: has_black_patch
This exercise demonstrates the data flow and computational logic of deep learning using a pure Python implementation. The goal is to build a model that can classify whether a provided grayscale image contains a 3x3 black square patch anywhere in the middle half of the image. For more information, refer to Chapter 2.

To train the `has_black_patch` model, use the following command:
```bash
deep-learning-at-scale chapter_2 has_black_patch train
```

Once the model is trained, you can visualize the embedding to explore the learned space and decision boundaries of the model. To do this, use the following command:
```bash
deep-learning-at-scale chapter_2 has_black_patch feature-embedding
```

### Profiling
To perform memory profiling, use the following commands:
```bash
mprof run deep-learning-at-scale chapter_2 has_black_patch train
mprof plot
```

To perform CPU profiling, use the following commands:
```bash
python -m cProfile -o output.pstats deep_learning_at_scale/chapter_2/has_black_patches_or_not.py train
gprof2dot --colour-nodes-by-selftime -f pstats output.pstats | dot -Tpng -o output.png
```

## Exercise 2: MNIST With PyTorch
This exercise involves a toy model implemented with PyTorch. To train this model, use the following command:
```bash
deep-learning-at-scale chapter_2 train
```

To inspect and explore the model using the torchinfo library, use the following command:
```bash
deep-learning-at-scale chapter_2 inspect-model
```

To visualize the computation graph generated for this toy problem, use the following command:
```bash
deep-learning-at-scale chapter_2 viz-model
```

### Visualize Loss Curvature
To visualize the simulated loss curvature for this model, use the following command:
```bash
deep-learning-at-scale chapter_2 simulate-loss-curve
```

### Inference Time Profiling
To convert the model to a Torch script and perform inference, use the following commands:
```bash
deep-learning-at-scale chapter_2 convert-to-torch-script
deep-learning-at-scale chapter_2 infer
```

## Exercise 1: has_black_patch
This exercise is a no-frills pure Python exercise that demonstrates the data flow and computational logic of deep learning. The `has_black_patch` exercise builds a model that can classify whether the provided grayscale image contains a 3x3 black square patch anywhere in the middle half of the image. For more information, please refer to Chapter 2.

To train the `has_black_patch` model, use the following command:
```bash
deep-learning-at-scale chapter_2 has_black_patch train
```

Once your model is trained, you can visualize the embedding to explore the learned space and decision boundaries of the model. To do this, use the following command:
```bash
deep-learning-at-scale chapter_2 has_black_patch feature-embedding
```

### Profiling
To perform memory profiling, use the following commands:
```bash
mprof run deep-learning-at-scale chapter_2 has_black_patch train
mprof plot
```

To perform CPU profiling, use the following commands:
```bash
python -m cProfile -o output.pstats deep_learning_at_scale/chapter_2/has_black_patches_or_not.py train
gprof2dot --colour-nodes-by-selftime -f pstats output.pstats | dot -Tpng -o output.png
```

## Exercise 2: MNIST With PyTorch
This exercise involves a toy model implemented with PyTorch. To train this model, use the following command:
```bash
deep-learning-at-scale chapter_2 train
```

To inspect and explore the model using the torchinfo library, use the following command:
```bash
deep-learning-at-scale chapter_2 inspect-model
```

To visualize the computation graph generated for this toy problem, use the following command:
```bash
deep-learning-at-scale chapter_2 viz-model
```

### Visualize Loss Curvature
To visualize the simulated loss curvature for this model, use the following command:
```bash
deep-learning-at-scale chapter_2 simulate-loss-curve
```

### Inference Time Profiling
To convert the model to a Torch script and perform inference, use the following commands:
```bash
deep-learning-at-scale chapter_2 convert-to-torch-script
deep-learning-at-scale chapter_2 infer
```

