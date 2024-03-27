import numpy as np

import copy

from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.util.visualize_mujoco import visualize_mujoco_from_states


class MujocoReplayBuffer(EnvReplayBuffer):

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

        self.body_xpos_shape = env.sim.data.body_xpos.shape
        self._body_xpos = np.zeros((max_replay_buffer_size, *self.body_xpos_shape))

        self.qpos_shape = env.sim.data.qpos.shape
        self._qpos = np.zeros((max_replay_buffer_size, *self.qpos_shape))

        self.env_states = []

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._body_xpos[self._top] = self.env.sim.data.body_xpos
        self._qpos[self._top] = self.env.sim.data.qpos
        if len(self.env_states) >= self.max_replay_buffer_size():
            self.env_states[self._top] = self.env.sim.get_state()
        else:
            self.env_states.append(copy.deepcopy(self.env.sim.get_state()))
        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(dict(
            body_xpos=self._body_xpos[:self._size],
            qpos=self._qpos[:self._size],
            env_states=self.env_states[:self._size],
        ))
        return snapshot

    def visualize_agent(self, start_idx, end_idx):
        visualize_mujoco_from_states(self.env, self.env_states[start_idx:end_idx])

    def reset(self):
        super().reset()

        self._body_xpos = np.zeros_like(self._body_xpos)
        self._qpos = np.zeros_like(self._qpos)

        self.env_states = []
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
        s_change = ((state_n) * (new_bservations - Smin)) / (np.array(Smax) - Smin)
        s_next_change = ((state_n) * (new_next_obs - Smin)) / (np.array(Smax) - Smin)
        s_change = (s_change + 0.5) // 1
        s_next_change = (s_next_change + 0.5) // 1
        s_change_cur = 0
        s_change_next = 0
        self._observation_dim = new_bservations.shape[1]
        for i in range(self._observation_dim):
            s_change_cur = (s_change[:, i] + (state_n + 1) * s_change_cur)
            s_change_next = (s_next_change[:, i] + (state_n + 1) * s_change_next)
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
        print('all_state_num', len(np.unique(state_num2)))
        return amin, amax, s_and_snext[0], s_and_snext[1], state_num2.max()
