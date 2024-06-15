# Chapter 9. Gaining Practical Expertise with Scaling Across All Dimensions



## Exercise 1: Baseline DeepFM

To execute this exercise, use the following command:

    ```bash
    deep-learning-at-scale chapter_9 pt-baseline-deepfm train
    ```


## Exercise 2: Model Parallel DeepFM

To execute this exercise, use the following command:

    ```bash
    deep-learning-at-scale chapter_9 pt-mp-deepfm train
    ```

## Exercise 3: Pipeline Parallel DeepFM

To execute this exercise, use the following command:

    ```bash
    deep-learning-at-scale chapter_9 pt-pipe-deepfm train
    ```


## Exercise 4: Pipeline Parallel DeepFM with RPC

To execute this exercise, use the following command:
    ```bash
    deep-learning-at-scale chapter_9 rpc-pt-pipeline-deepfm train
    ```


## Exercise 5: Tensor Parallel DeepFM

To execute this exercise, use the following command:
    ```bash
    deep-learning-at-scale chapter_9 pt-tensor-deepfm train --use-pairwise-parallel
    ```


## Exercise 6: Hybrid Parallel DeepFM
To execute this exercise, use the following command:
    ```bash
    deep-learning-at-scale chapter_9 pt-tensor-ddp-deepfm train
    ```

`Note`: The exercises here were originally targeted for PyTorch 2.1. FSDP has evolved rapidly since PyTorch 2.2. As a result some of the FSDP examples may be outdated. These examples will be updated for PT2.4 soon.

## Exercise 7: Automatic Vertical Scaling with DeepSpeed

To execute this exercise, use the following command:
    ```bash
    deep-learning-at-scale chapter_9 zero3 train
    ```
