# MineRL RLlib Benchmark [WIP]

Here we benchmark various reinforcement learning algorithms available in [RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) on the [MineRL](https://minerl.io/docs/) environment.

[RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. 
RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic.

## Installation

1. Make sure you have JDK 1.8 on your system for [MineRL](https://minerl.io/docs/tutorials/index.html#installation).
2. Install [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install) with correct cuda version
3. Install [Ray Latest Snapshots (Nightlies)](https://docs.ray.io/en/master/installation.html#latest-snapshots-nightlies) 
4. Finally, 
```
git clone https://github.com/juliusfrost/minerl-rllib.git
cd minerl-rllib
pip install -r requirements.txt
```

### Parallel Environments

Multiprocessing is available but not merged into minerl ([see this pull](https://github.com/minerllabs/minerl/pull/352)). For now, do
```
git clone https://github.com/minerllabs/minerl.git
cd minerl
git fetch origin pull/352/head:process-safe
git checkout process-safe
pip install -e .
```

## Implementation Details
See [Planned Implementation Details](Implementation.md)

For discrete action spaces, make sure you have the environment variable `MINERL_DATA_ROOT` set, 
otherwise it defaults to the `data` folder. 
Then, run `python generate_kmeans.py`

### Configuration

This repository comes with a modular configuration system.
We specify configuration `yaml` files according to the rllib specification.
You can see an example config at `config/minerl-impala-debug.yaml`

### Training
We use `rllib_train.py` to train our RL algorithm on MineRL.
The easiest way to get started is to specify all of your configurations first as stated in the previous section.
See help with `python rllib_train.py --help`

Here is an example command:
```
python rllib_train.py -f config/minerl-impala-debug.yaml
```
This will run the IMPALA RL algorithm on the MineRL Debug Environment,
which has the same observation space and action space as the competition environments 
but doesn't run Minecraft explicitly.

### Testing
See help with `python rllib_test.py --help`
