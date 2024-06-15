# Chapter 4. Putting It All Together—Efficient Deep Learning

This chapter builds on the content from the previous two chapters, demonstrating the acceleration provided by specialized computing hardware and providing some examples of how-tos. It also presents some tips and tricks for efficiently training a deep learning model on a single machine with at most one accelerated device.

There are two hands-on exercises in this chapter, one using a language model (OpenAI’s GPT-2) and the second an image classification model (EfficientNet).

## Exercise 1: GPT-2

To execute this exercise, use the following command:

```bash
deep-learning-at-scale chapter_4 train_gpt2
```

Ensure that your experiment tracking server, [aimhub](https://github.com/aimhubio/aim) in this case, is already running. For more info on this, see [this section](../../README.md#setting-up-for-experiment-tracking-with-aimhub)

## Exercise 2: Image Classification

To execute this exercise, use the following command:

```bash
deep-learning-at-scale chapter_4 train_efficient_unet
```