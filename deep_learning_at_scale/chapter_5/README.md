# Chapter 5: Distributed Systems and Communications

This chapter explores the different types of distributed systems and the challenges they present. It also covers various communication topologies and techniques that are used in enabling deep learning in a distributed setting. Additionally, this chapter provides an overview of software and frameworks that can help manage processes and infrastructure at scale.

## Exercise 1: MPI

To execute this exercise, use the following commands:

```bash
mpicc -o probe ./deep_learning_at_scale/chapter_5/mpi_probe.cpp
mpirun -np 2 ./probe
mpirun -np 3 -H localhost:1,localhost:2,localhost:3 ./probe
```

## Exercise 2: Collective communication in PyTorch:

```bash
python deep_learning_at_scale/chapter_5/torch_dist.py --no-use-async 
--world-size 3
```