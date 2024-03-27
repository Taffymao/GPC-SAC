from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning.gpc_sac import PEVITrainer
import lifelong_rl.util.pythonplusplus as ppp
from lifelong_rl.models.networks import FlattenMlp ,RandomPrior
import os
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.nn import functional as F


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):
    """
    Policy construction
    """

    num_qs = variant['trainer_kwargs']['num_qs']
    M = variant['policy_kwargs']['layer_size']
    num_q_layers = variant['policy_kwargs']['num_q_layers']
    num_p_layers = variant['policy_kwargs']['num_p_layers']
    qfs, target_qfs = [], []
    for _ in range(num_qs):
        qfn = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M] * num_q_layers)
        target_qfn = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M] * num_q_layers)
        qfs.append(qfn)
        target_qfs.append(target_qfn)
    """
          TanhGaussianPolicy Usage:

          ```
          policy = TanhGaussianPolicy(...)
          action, mean, log_std, _ = policy(obs)
          action, mean, log_std, _ = policy(obs, deterministic=True)
          action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
          ```

          Here, mean and log_std are the mean and log_std of the Gaussian that is
          sampled from.

          If deterministic is True, action = tanh(mean).
          If return_log_prob is False (default), log_prob = None
              This is done because computing the log_prob can be a bit expensive.
          """
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_p_layers,
        layer_norm=None,
    )


    """
      Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
      Continuous maximum Q-learning algorithm with parameterized actor.
      """
    trainer = PEVITrainer(
        env=eval_env,
        policy=policy,
        qfs=qfs,
        target_qfs=target_qfs,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs'],
    )
    """
    Create config dict
    """

    config = dict()
    config.update(
        dict(
            trainer=trainer,
            exploration_policy=policy,
            evaluation_policy=MakeDeterministic(policy),
            exploration_env=expl_env,
            evaluation_env=eval_env,
            replay_buffer=replay_buffer,
            qfs=qfs,
        ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())

    return config
