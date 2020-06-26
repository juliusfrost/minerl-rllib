# MineRL RLlib Benchmark [WIP]

Here we benchmark various reinforcement learning algorithms available in [RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) on the [MineRL](https://minerl.io/docs/) environment.

[RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. 
RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic.

## Installation

OS: Linux (waiting on ray Windows support)

Make sure you have JDK 1.8 on your system for MineRL.

Install [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install) with correct cuda version, then
`pip install -r requirements.txt`

## Experiments

Currently experiments do not support headless mode.

### SAC [WIP]

`python tune.py`
