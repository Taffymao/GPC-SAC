import gym
from gym.spaces import Discrete

from lifelong_rl.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from lifelong_rl.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space

        self._action_space = env.action_space
        self._meta_infos = []

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        if isinstance(self._ob_space, gym.spaces.Box):
            self._ob_shape = self._ob_space.shape
        else:
            self._ob_shape = None

        super().__init__(max_replay_buffer_size=max_replay_buffer_size,
                         observation_dim=get_dim(self._ob_space),
                         action_dim=get_dim(self._action_space),
                         env_info_sizes=env_info_sizes)

    def obs_preproc(self, obs):
        if len(obs.shape) > len(self._ob_space.shape):
            obs = np.reshape(obs, (obs.shape[0], self._observation_dim))
        else:
            obs = np.reshape(obs, (self._observation_dim, ))
        return obs

    def obs_postproc(self, obs):
        if self._ob_shape is None:
            return obs
        if len(obs.shape) > 1:
            obs = np.reshape(obs, (obs.shape[0], *self._ob_shape))
        else:
            obs = np.reshape(obs, self._ob_shape)
        return obs

    def add_sample(self,
                   observation,
                   action,
                   reward,
                   terminal,
                   next_observation,
                   env_info=None,
                   **kwargs):
        if hasattr(self.env, 'get_meta_infos'):
            self._meta_infos.append(self.env.get_meta_infos())
        if env_info is None:
            env_info = dict()
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(observation=observation,
                                  action=new_action,
                                  reward=reward,
                                  next_observation=next_observation,
                                  terminal=terminal,
                                  env_info=env_info,
                                  **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot['meta_infos'] = self._meta_infos
        return snapshot

    def reset(self):
        super().reset()

        self._meta_infos = []

    def state_transform(self, state_n):

        s1min = (np.min(self._observations, axis=0))
        s1max = (np.max(self._observations, axis=0))
        s2min = (np.min(self._next_obs, axis=0))
        s2max = (np.max(self._next_obs, axis=0))
        amin = (np.min(self._actions, axis=0))
        amax = (np.max(self._actions, axis=0))
        smin = np.minimum(s1min, s2min)
        smax = np.maximum(s1max, s2max)
        Smax = []
        Smin = []
        j = []
        state_n = int(state_n)
        for i in range(self._observation_dim):
            if smax[i] - smin[i] != 0:
                Smin.append(smin[i])
                Smax.append(smax[i])
            else:
                j.append(i)
        new_bservations = np.delete(self._observations, j, axis=1)
        new_next_obs = np.delete(self._next_obs, j, axis=1)
        s_change = ((state_n ) * (new_bservations - Smin)) / (np.array(Smax) - Smin)
        s_next_change = ((state_n ) * (new_next_obs - Smin)) / (np.array(Smax) - Smin)
        s_change = (s_change + 0.5) // 1
        s_next_change = (s_next_change + 0.5) // 1
        s_change_cur = 0
        s_change_next = 0
        self._observation_dim = new_bservations.shape[1]
        for i in range(self._observation_dim):
            s_change_cur = (s_change[:, i] + (state_n+1) * s_change_cur)
            s_change_next = (s_next_change[:, i] + (state_n+1) * s_change_next)
        r = 1
        state_num = np.append(s_change_cur, s_change_next)
        state_num2_sort = state_num2 = np.zeros_like(state_num)
        order = np.argsort(state_num)
        state_num_sort = sorted(state_num)
        for i in range(2 * self._size - 1):
            j = i + 1
            if state_num_sort[j] == state_num_sort[i]:
                state_num2_sort[j] = state_num2_sort[i]
            else:
                state_num2_sort[j] = r
                r = r + 1
        state_num2_sort = sorted(state_num2_sort)
        for m in range(len(state_num)):
            state_num2[order[m]] = state_num2_sort[m]
        s_and_snext = np.split(state_num2, 2)
        print('all_state_num',len(np.unique(state_num2)))
        return amin, amax, s_and_snext[0], s_and_snext[1], state_num2.max()
