# MineRL RLlib Benchmark

Here we benchmark various reinforcement learning algorithms available in [RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) on the [MineRL](https://minerl.io/docs/) environment.

[RLlib](https://docs.ray.io/en/master/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. 
RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic.

## Installation

Make sure you have JDK 1.8 on your system for [MineRL](https://minerl.io/docs/tutorials/index.html#installation)

Requires Python 3.7 or 3.8.

#### Use a conda virtual environment
```bash
conda create --name minerl-rllib python=3.8
conda activate minerl-rllib
```

#### Install dependencies
```bash
pip install poetry
poetry install
```
Install [PyTorch](https://pytorch.org/get-started/locally/) with correct cuda version.

## How to Use
### Data
Make sure you have the environment variable `MINERL_DATA_ROOT` set, 
otherwise it defaults to the `data` folder.

#### Downloading the MineRL dataset
Follow the official instructions: https://minerl.io/dataset/  
If you download the data to `./data` then you don't need to set `MINERL_DATA_ROOT` in your environment variables.

### Training
Do `train.py --help` to see all options.
See the following command trains the SAC algorithm on offline data in the `MineRLObtainDiamondVectorObf-v0` environment.
```
python train.py -f config/sac-offline.yaml
```

#### Configuration

This repository comes with a modular configuration system.
We specify configuration `yaml` files according to the rllib specification.
