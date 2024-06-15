# Chapter 11. Scaling Experiments: Effective Planning and Management


## Exercise 1: Transfer Learning

Please see [chapter_4](../chapter_4/vision_model.py) for this.

## Exercise 2: Hyperparameter Optimization

1. Create postgres database:
    ```bash
    docker run --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres
    docker exec -it postgres bash
    CREATE DATABASE study_db;
    ```
2. Run HPO exercise
   ```bash
   deep-learning-at-scale chapter_11 hpo train
   ```
3. Explore Optuna dashboard
   ```bash
   optuna-dashboard postgresql+psycopg2://postgres:postgres@hostname:5432/study_db
   ```

## Exercise 3: Knowledge Distillation
    ```bash
    deep-learning-at-scale chapter_11 distill train
    ```

Train student baseline:
    ```bash
    deep-learning-at-scale chapter_11 distill train-student-baseline
    ```


## Exercise 4: Mixture of Experts
    ```bash
    deep-learning-at-scale chapter_11 moe run
    ```

### With deepspeed MoE:
Create baseline using:
    ```bash
    deep-learning-at-scale chapter_11 mnist-baseline train
    ```
Run baseline using:
    ```bash   
    deepspeed \
        --num_nodes=1 \
        --num_gpus=2 \
        --bind_cores_to_rank \
        deep_learning_at_scale/chapter_11/mnist_deepspeed.py
    ```

## Exercise #5: Contrastive Learning

This exercise is a light weight immplementation  of `SimCLR: a simple framework for contrastive learning` [1,2].  

    ```bash
    deep-learning-at-scale chapter_11 cl train
    ```



### References:

1. [Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ArXiv. /abs/2002.05709](https://arxiv.org/abs/2002.05709)

2. [Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. (2020). Big Self-Supervised Models are Strong Semi-Supervised Learners. ArXiv. /abs/2006.10029](https://arxiv.org/abs/2006.10029)