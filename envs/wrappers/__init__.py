from envs.wrappers.action_repeat_wrapper import MineRLActionRepeat
from envs.wrappers.action_wrapper import MineRLActionWrapper
from envs.wrappers.deterministic_wrapper import MineRLDeterministic
from envs.wrappers.discrete_action_wrapper import MineRLDiscreteActionWrapper
from envs.wrappers.gray_scale_wrapper import MineRLGrayScale
from envs.wrappers.normalize import MineRLNormalizeObservationWrapper, MineRLNormalizeActionWrapper, \
    MineRLRewardScaleWrapper
from envs.wrappers.observation_stack_wrapper import MineRLObservationStack
from envs.wrappers.observation_wrapper import MineRLObservationWrapper
from envs.wrappers.time_limit_wrapper import MineRLTimeLimitWrapper


def wrap(env, discrete=False, num_actions=32, data_dir=None, num_stack=1, action_repeat=1,
         gray_scale=False, seed=None, normalize_observation=False, normalize_action=False, reward_scale=1., **kwargs):
    env = MineRLTimeLimitWrapper(env)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    if discrete:
        env = MineRLDiscreteActionWrapper(env, num_actions, data_dir=data_dir)
    if gray_scale:
        env = MineRLGrayScale(env)
    if normalize_observation:
        env = MineRLNormalizeObservationWrapper(env, kwargs.get('norm_pov_low', 0.), kwargs.get('norm_pov_high', 1.),
                                                kwargs.get('norm_vec_low', -1.), kwargs.get('norm_vec_high', 1.))
    if normalize_action:
        if discrete:
            print('Tried to normalize discrete actions which is not possible! '
                  'Skipping the normalizing action wrapper.')
        else:
            env = MineRLNormalizeActionWrapper(env, kwargs.get('norm_act_low', -1.), kwargs.get('norm_act_high', 1.))
    if reward_scale != 1.:
        env = MineRLRewardScaleWrapper(env, reward_scale)
    if num_stack > 1:
        env = MineRLObservationStack(env, num_stack)
    if action_repeat > 1:
        env = MineRLActionRepeat(env, action_repeat)
    if seed is not None:
        env = MineRLDeterministic(env, seed)
    return env
