import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.utils import merge_dicts

from minerl_rllib import models

BASELINE_CONFIG = {
    # specifies the size of the state embedding
    # this is the rnn hidden size if use_rnn = True
    'state_embed_size': 512,
    # specifies whether to use a recurrent neural network for the state embedding
    'use_rnn': True,
    # specifies the type of rnn layer
    # implemented: [gru, lstm]
    'rnn_type': 'gru',
    # specifies whether to include the previous action and reward as part of the observation
    'use_prev_action_reward': False,
    # specifies extra key word arguments for the RNN class
    'rnn_config': {
        # specifies the rnn hidden layer size
        'hidden_size': 512,
        # specifies the number of rnn layers
        'num_layers': 1,
    },

    # specifies the network architecture for pov image observations
    # gets the pov network factory method from pov.py
    # takes in pov_net_kwargs as input and produces pov_network and pov_embed_size
    'pov_net': 'baseline',
    # specifies the network architecture for vector observations
    # gets the vector network factory method from vector.py
    # takes in vector_net_kwargs as input and produces vector_network and vector_embed_size
    'vector_net': 'baseline',
    # specifies the network architecture for action observations (discrete or continuous)
    # gets the action network factory method from action.py
    # takes in action_net_kwargs as input and produces action_network and action_embed_size
    'action_net': 'baseline',
    # specifies the network architecture for reward observations
    # gets the reward network factory method from reward.py
    # takes in reward_net_kwargs as input and produces reward_network and reward_embed_size
    'reward_net': 'baseline',

    # key word arguments for the pov network factory method
    'pov_net_kwargs': {
        'pov_embed_size': 256
    },
    # key word arguments for the vector network factory method
    'vector_net_kwargs': {
        'vector_embed_size': 512
    },
    # key word arguments for the action network factory method
    'action_net_kwargs': {
        'action_embed_size': 128
    },
    # key word arguments for the reward network factory method
    'reward_net_kwargs': {
        'reward_embed_size': 128
    },
}


class MineRLTorchModel(TorchModelV2, nn.Module):
    """
    Baseline MineRL PyTorch Model
    See rllib documentation on custom models: https://docs.ray.io/en/master/rllib-models.html#pytorch-models
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        model_config = merge_dicts(BASELINE_CONFIG, model_config['custom_model_config'])
        # new way to get model config directly from keyword arguments
        model_config = merge_dicts(model_config, kwargs)

        state_embed_size = model_config['state_embed_size']
        self.use_rnn = model_config['use_rnn']
        rnn_type = model_config['rnn_type']
        self.use_prev_action_reward = model_config['use_prev_action_reward']

        action_net_kwargs = model_config['action_net_kwargs']
        if isinstance(action_space, gym.spaces.Discrete):
            action_net_kwargs.update({'discrete': True, 'n': action_space.n})
            self.discrete = True
        else:
            self.discrete = False

        def get_factory(network_name):
            return getattr(getattr(models.torch, network_name), model_config[f'{network_name}_net'])

        self._pov_network, pov_embed_size = get_factory('pov')(**model_config['pov_net_kwargs'])
        self._vector_network, vector_embed_size = get_factory('vector')(**model_config['vector_net_kwargs'])
        state_input_size = pov_embed_size + vector_embed_size
        if self.use_prev_action_reward:
            self._action_network, action_embed_size = get_factory('action')(**action_net_kwargs)
            self._reward_network, reward_embed_size = get_factory('reward')(**model_config['reward_net_kwargs'])
            state_input_size += action_embed_size + reward_embed_size

        rnn_config = model_config.get('rnn_config')
        if self.use_rnn:
            state_embed_size = rnn_config['hidden_size']
            if rnn_type == 'lstm':
                self._rnn = minerl_rllib.models.torch.rnn.LSTMBaseline(state_input_size, **rnn_config)
            elif rnn_type == 'gru':
                self._rnn = minerl_rllib.models.torch.rnn.GRUBaseline(state_input_size, **rnn_config)
            else:
                raise NotImplementedError
        else:
            self._state_network = nn.Sequential(
                nn.Linear(state_input_size, state_embed_size),
                nn.ELU(),
            )
        self._value_head = nn.Sequential(
            nn.Linear(state_embed_size, 1),
        )
        self._policy_head = nn.Sequential(
            nn.Linear(state_embed_size, num_outputs),
        )

    def get_initial_state(self):
        if self.use_rnn:
            return self._rnn.initial_state()
        return []

    def forward(self, input_dict, state, seq_lens):
        recursive_print(dict(input_dict=input_dict, state=state, seq_lens=seq_lens))
        includes_time_dim = False if seq_lens is None else len(seq_lens) > 0

        flat_shape = input_dict['obs_flat'].shape
        if len(flat_shape) == 3:
            batch_t = flat_shape[0]
            batch_n = flat_shape[1]
            raise ValueError(f"obs_flat shape = {flat_shape}")
        elif len(flat_shape) == 2:
            if includes_time_dim:
                batch_n = len(seq_lens)
                batch_t = flat_shape[0] // batch_n
            else:
                batch_t = 1
                batch_n = flat_shape[0]
        else:
            batch_t = 1
            batch_n = 1

        device = next(self.parameters()).device
        pov = input_dict['obs'][0].permute(0, 3, 1, 2)  # n,c,h,w
        # if len(input_dict['obs'][0].shape) == 4:
        #     pov = input_dict['obs'][0].permute(0, 3, 1, 2)  # n,c,h,w
        # elif len(input_dict['obs'][0].shape) == 5:
        #     pov = input_dict['obs'][0].permute(0, 1, 4, 2, 3)  # t,n,c,h,w
        # else:
        #     raise ValueError('expected different shape for pov input')
        vector = input_dict['obs'][1]
        pov_embed = self._pov_network(pov.to(device))
        pov_embed = torch.reshape(pov_embed, pov_embed.shape[:2])
        vector_embed = self._vector_network(vector.to(device))

        if self.use_prev_action_reward:
            prev_actions = input_dict['prev_actions']
            if self.discrete:
                prev_actions = prev_actions.long()
            prev_rewards = input_dict['prev_rewards']

            action_embed = self._action_network(prev_actions.to(device))
            reward_embed = self._reward_network(prev_rewards.to(device))

            state_inputs = torch.cat((pov_embed, vector_embed, action_embed, reward_embed), dim=-1)
        else:
            state_inputs = torch.cat((pov_embed, vector_embed), dim=-1)

        if self.use_rnn:
            state_size = state_inputs.shape[-1]
            state_inputs = torch.reshape(state_inputs, (batch_t, batch_n, state_size))
            # if includes_time_dim:
            #     state_inputs = torch.reshape(state_inputs, (batch_t, batch_n, state_size))
            #     print(state_inputs, state_inputs.shape, seq_lens)
            #     state_inputs = torch.nn.utils.rnn.pack_padded_sequence(state_inputs, seq_lens, enforce_sorted=False)
            # else:
            #     state_inputs = torch.reshape(state_inputs, (batch_t, batch_n, state_size))
            rnn_output, rnn_state = self._rnn(state_inputs, state)
            # if includes_time_dim:
            #     rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)
            self._logits = torch.reshape(rnn_output, (batch_t * batch_n, self._rnn.hidden_size))
        else:
            self._logits = self._state_network(state_inputs)
            rnn_state = []

        outputs = self._policy_head(self._logits)
        return outputs, rnn_state

    def value_function(self):
        values = self._value_head(self._logits)
        return values

    def import_from_h5(self, h5_file):
        raise NotImplementedError

    def custom_loss(self, policy_loss, loss_inputs):
        if self.use_rnn:
            return [policy_loss[0] + self._rnn.custom_loss()]
        return policy_loss


def register():
    ModelCatalog.register_custom_model('minerl_torch_model', MineRLTorchModel)


def recursive_print(d):
    if isinstance(d, dict):
        print('{')
        for k, v in d.items():
            print(f'key={k}')
            recursive_print(v)
        print('}')
    elif isinstance(d, list) or isinstance(d, tuple):
        print('[')
        for v in d:
            recursive_print(v)
        print(']')
    elif isinstance(d, np.ndarray) or isinstance(d, torch.Tensor):
        print(d.shape)
    else:
        print(d)
