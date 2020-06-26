import os
import gym
import ray
import minerl
from ray import tune
# from ray.rllib.agents import a3c
# from ray.tune.registry import register_env
# from ray.tune.logger import pretty_print

import register_env

env_name = "MineRLObtainDiamondDenseVectorObf-v0"
# offline_path = os.path.join(os.environ['MINERL_DATA_ROOT'], 'rllib', env_name, '*.json')

ray.init()
tune.run(
    "SAC",
    stop={"time_total_s": int(60 * 60 * 2)},
    config={
        "env": env_name,
#         "input": {"sampler": 0.5, offline_path: 0.5},
#         "input_evaluation": ["simulation"],
        "explore": False,
        "num_gpus": 1,
        "num_workers": 0,
#         "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#         "eager": False,
        "use_pytorch": True,
#         "framework": "torch",
        "eager_tracing": True,
        "gamma": 0.99,
        "use_state_preprocessor": True,
        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [512],
        },
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [512],
        },
        "tau": 1.0,
        "target_network_update_freq": 8000,
        "target_entropy": "auto",
        "clip_rewards": 1.0,
        "no_done_at_end": False,
        "n_step": 1,
        "rollout_fragment_length": 1,
        "prioritized_replay": True,
        "train_batch_size": 64,
        "timesteps_per_iteration": 4,
        "learning_starts": 20000,
        "optimization": {
            "actor_learning_rate": 0.0003,
            "critic_learning_rate": 0.0003,
            "entropy_learning_rate": 0.0003,
        },
        "metrics_smoothing_episodes": 5,
    },
)
    
# ray.init()
# config = a3c.DEFAULT_CONFIG.copy()
# config['env_config'] = dict()
# config['num_workers'] = 0
# config['num_envs_per_worker'] = 1
# config['num_gpus'] = 1
# trainer = a3c.A2CTrainer(config, env='MineRLObtainDiamondDenseVectorObf-v0')

# for i in range(1000):
#     result = trainer.train()
#     print(pretty_print(result))

#     if i % 100 == 0:
#         checkpoint = trainer.save()
#         print("checkpoint saved at", checkpoint)
