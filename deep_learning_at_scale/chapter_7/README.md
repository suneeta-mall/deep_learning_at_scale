# Chapter 7: Data Parallelism

## Distributed Sampler

To use the distributed sampler, run the following command:

```bash
deep-learning-at-scale chapter_7 distribute-iterable start-worker
```

## Exercise 1: RPC Centralized Distributed Training

To perform RPC centralized distributed training, use the following command:

```bash
deep-learning-at-scale chapter_7 rpc train --world-size 2
```

## Exercise 2: Centralized Gradient-Partitioned Joint Worker/Server Distributed Training

To perform centralized gradient-partitioned joint worker/server distributed training, use the following command:

```bash
deep-learning-at-scale chapter_7 ddp-centralized train --world-size 1
```

## Exercise 3: Decentralized Asynchronous Distributed Training

To perform decentralized asynchronous distributed training, use the following command:

```bash
deep-learning-at-scale chapter_7 hogwild train --world-size 4
```

## Exercise 4: Scene Parsing with DDP

To perform scene parsing with DDP, use the following command:

```bash
deep-learning-at-scale chapter_7 ddp train
```

Multinode:

To perform multinode training, use the following command:

```bash
MASTER_PORT=<xxx> MASTER_ADDR=<yyy> deep-learning-at-scale chapter_7 ddp train 
–nodes 2
```

## Exercise 5: Distributed Sharded DDP (ZeRO)

To perform distributed sharded DDP (ZeRO), use the following command:

```bash
deep-learning-at-scale chapter_7 sharded-ddp train \
    --precision 16-mixed \
    --strategy deepspeed_stage_1
```

## Exercise 6: Special Instruction For FFCV setup.

This exercise includes instructions for setting up FFCV with optimized image processing APIs. To set up FFCV, follow the instructions in the [Dockerfile extra](../../Dockerfile.extra).

1. Data prep:

To prepare the data, run the following command:

```bash
deep-learning-at-scale chapter_7 efficient-ddp data-to-beton
```

2. Train:

To train the model, run the following command:

```bash
deep-learning-at-scale chapter_7 efficient-ddp train

# deep-learning-at-scale chapter_7 efficient-ddp train \
#    --precision 16-mixed \
#    --batch-size 105
```
