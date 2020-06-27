# MineRL RLlib Baselines (Planned) Implementation

## Environments
### Treechop
1. `MineRLTreechopVectorObf-v0`

### Navigate
1. `MineRLNavigateVectorObf-v0`
2. `MineRLNavigateExtremeVectorObf-v0`
3. `MineRLNavigateDenseVectorObf-v0`
4. `MineRLNavigateExtremeDenseVectorObf-v0`

### Obtain Diamond
1. `MineRLObtainDiamondVectorObf-v0`
2. `MineRLObtainDiamondDenseVectorObf-v0`

### Obtain Iron Pickaxe
We will ignore this because it is a task subset of Obtain Diamond.
However using this category human data may be valuable.
1. `MineRLObtainIronPickaxeVectorObf-v0`
2. `MineRLObtainIronPickaxeDenseVectorObf-v0`

## Metrics
1. Final episode reward
2. Final episode reward with human normalized performance
3. Sample efficiency (0, 100k, 500k, 1M, 8M)
4. Episode reward curves

## Action Space
1. Continuous
    1. Naturally supported by minerl vector actions (64,)
2. Discrete
    1. [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) on human data and use as discretized action space
    2. [Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) to map data to discrete actions

## Observation Space
1. Tuple Observations (Image (64,64,3), vector (64,))

## Model
1. Convolutional Neural Network for image observations
2. Concatenate hidden output with vector observations
3. Feed concatenation into feed-forward network (and possibly RNN) for latent state representation
4. Use latent state representation for policy, value, Q networks.

## RL setting

### Online
1. Online exploration
    1. Use algorithm default exploration
2. Learn policy from environment sampled data
3. Both on-policy and off-policy RL algorithms

### Offline
1. No exploration
2. Learn policy from human data
3. Only off-policy RL algorithms

### Mixed
1. Online exploration in the environment
2. Learn from environment sampled data and human data
3. Only off-policy RL algorithms
