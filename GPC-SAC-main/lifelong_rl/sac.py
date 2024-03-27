import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch

# from torch_batch_svd import svd

ACTION_MIN = -1.0
ACTION_MAX = 1.0


class SACTrainer(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            reward_scale=1.0,  # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            qf_lr=3e-4,  # Learning rate of Q functions
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            soft_target_tau=5e-3,  # Rate of update of target networks
            target_update_period=1,  # How often to update target networks
            #与sac的差别
            max_q_backup=False,
            deterministic_backup=False,
            policy_eval_start=0,
            eta=-1.0,

            num_qs=10,
            replay_buffer=None,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qs = num_qs

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta

        self.replay_buffer = replay_buffer

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

    def _get_tensor_values(self, obs, actions, network=None):  #sac中没有的定义
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat,
                                           1).view(obs.shape[0] * num_repeat,
                                                   obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(-1, obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):#sac中没有的定义
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions,
                                           1).view(obs.shape[0] * num_actions,
                                                   obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp,
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach() #detach 意为分离，对某个张量调用函数 d e t a c h ( ) \rm detach()detach() 的作用是返回一个 T e n s o r \rm TensorTensor，它和原张量的数据相同，但 r e q u i r e s _ g r a d = F a l s e \rm requires\_grad=Falserequires_grad=False，也就意味着 d e t a c h ( ) \rm detach()detach() 得到的张量不会具有梯度
    def train_from_torch(self, batch):
        obs= batch['observations']
        next_obs = batch['next_observations']#obs和next_obs一个生成标签一个用于实在动作
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        if self.eta > 0:
            actions.requires_grad_(True)      #用于说明当前量是否需要在计算中保留对应的梯度信息
        
        """
        Policy and Alpha Loss           
        """

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(   #有所不同，在当前的情况下根据策略从数据集这里选一个动作可以说是采了个odd样本
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = self.qfs.sample(obs, new_obs_actions)  #根据Q值的ensemble来采用动作？

        policy_loss = (alpha * log_pi - q_new_actions).mean()#伪代码第4步，最大化最小动作的Q值减去熵

        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.get_log_probs(obs.detach(), actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()#伪代码第6步，actor loss

            
        """
        QF Loss
        """
        # (num_qs, batch_size, output_size)
        qs_pred = self.qfs(obs, actions)
        print(qs_pred.size())

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=False,
            return_log_prob=True,
        )

        if not self.max_q_backup:
            target_q_values = self.target_qfs.sample(next_obs, new_next_actions)#从多个中采样一个，应该是找q最小的
            if not self.deterministic_backup:
                target_q_values -= alpha * new_log_pi
        else:
            # if self.max_q_backup
            next_actions_temp, _ = self._get_policy_actions(                #进行10步之后的动作和观测作为标签，只使用一次吗
                next_obs, num_actions=10, network=self.policy)
            #返回的next_actions_temp是一个值
            target_q_values = self._get_tensor_values(
                next_obs, next_actions_temp,
                network=self.qfs).max(2)[0].min(0)[0]
        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values
        qfs_loss = self.qf_criterion(qs_pred, q_target.detach().unsqueeze(0))#单个的loss
        qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()

        qfs_loss_total = qfs_loss
        
        if self.eta > 0:         #EDAC的不同
            obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)#requires_grad_用于说明当前量是否需要在计算中保留对应的梯度信息
            qs_preds_tile = self.qfs(obs_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            qs_pred_grads = qs_pred_grads.transpose(0, 1)
            
            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)#矩阵求积
            masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)#生成对角线全1，其余部分全0的二维数组
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qs - 1)
            
            qfs_loss_total += self.eta * grad_loss#伪代码第5步中的蓝色部分

        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        self.try_update_target_networks()#更新qfs，target-qfs等
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()

            self.eval_statistics['QFs Loss'] = np.mean(
                ptu.get_numpy(qfs_loss)) / self.num_qs
            if self.eta > 0:
                self.eval_statistics['Q Grad Loss'] = np.mean(
                    ptu.get_numpy(grad_loss))
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Predictions',
                    ptu.get_numpy(qs_pred),
                ))
            print(ptu.get_numpy(qs_pred))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Targets',
                    ptu.get_numpy(q_target),
                ))

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            qfs_optim=self.qfs_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
