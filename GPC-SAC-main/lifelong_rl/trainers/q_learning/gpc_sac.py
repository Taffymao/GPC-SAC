
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import math
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.gpc_torch_rl_algorithm import TorchTrainer
device = torch.device("cuda")


class PEVITrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qfs,
            target_qfs,
            action_n=18,
            beta=3,
            state_n=18,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            num_qs=2,
            prior=False,

            # CQL   Conservative Q-Learning
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            # sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            with_lagrange=False,
            lagrange_thresh=0.0,
            replay_buffer=None,
            target_update_period=1,  # How often to update target networks
            eta=-1,
    ):
        super().__init__()

        self.target_update_period = target_update_period
        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.soft_target_tau = soft_target_tau  # 0.005
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.replay_buffer = replay_buffer
        self.action_n=action_n
        self.state_n=state_n
        self.beta = beta
        # define an optimizer for log_alpha. The initial value of log_alpha is 0.
        if self.use_automatic_entropy_tuning:  # True
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)  # [0.]
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)  # policy_lr=0.0001

        self.with_lagrange = with_lagrange  # True or False
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # 5.0
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)  # [0.]
            # Optimizer for log_alpha_prime
            self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)  # qf_lr=0.0003

        self.plotter = plotter  # None
        self.render_eval_paths = render_eval_paths  # False

        self.qf_criterion_all = nn.MSELoss(reduction='none')
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.eta = eta
        self.discount = discount  # 0.99
        self.reward_scale = reward_scale  # 1
        self.eval_statistics = OrderedDict()  # dict
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._current_pre_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        self.temp = temp  # 1.0
        self.min_q_version = min_q_version  # 3
        self.min_q_weight = min_q_weight  # 10.0

        self.softmax = torch.nn.Softmax(dim=1)  #
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)
        self.max_q_backup = max_q_backup  # False
        self.deterministic_backup = deterministic_backup  # True

        self.discrete = False

        # ucb
        self.num_qs = num_qs
        self.prior = prior

        # Define optimizer for critic and actor
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.qfs_optimizer = []
        for i in range(self.num_qs):  # each ensemble member has its optimizer
            self.qfs_optimizer.append(optimizer_class(self.qfs[i].parameters(), lr=qf_lr))
        self.amin, self.amax, self.s_curr, self.s_next,s_size = self.replay_buffer.state_transform(self.state_n)
        if action_n%1 ==0:
            self.num_sa = torch.empty([int(s_size+1),int(((action_n+1) ** (self.replay_buffer._action_dim)))])
        else:
            self.num_sa = torch.empty([int(s_size+1),int(4*((action_n+1) ** (self.replay_buffer._action_dim)))])

    def func(self, obs, act, mean=False):
        # Using the main-Q network to calculate the bootstrapped uncertainty
        # Sample 10 ood actions for each obs, so the obs should be expanded before calculating
        action_shape = act.shape[0]          #2560
        obs_shape = obs.shape[0]                #256
        num_repeat = int(action_shape / obs_shape)  #10
        if num_repeat != 1:
            obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat,
                                                                 obs.shape[1])  # （2560, obs_dim）
        # Bootstrapped uncertainty
        qs_pred = []
        for i in range(self.num_qs):
            qs_pred.append(self.qfs[i](obs.cuda(), act.cuda()))
        if mean:
            qs_pred = torch.mean(torch.hstack(qs_pred), dim=1, keepdim=True)

        return qs_pred

    def func_target(self, obs_next, act_next):
        # Using the target-Q network to calculate the bootstrapped uncertainty
        # Sample 10 ood actions for each obs, so the obs should be expanded before calculating
        action_shape = act_next.shape[0]  # 2560
        obs_shape = obs_next.shape[0]  # 256
        num_repeat = int(action_shape / obs_shape)  # 10
        if num_repeat != 1:
            obs_next = obs_next.unsqueeze(1).repeat(1, num_repeat, 1).view(obs_next.shape[0] * num_repeat,
                                                                           obs_next.shape[1])  # （2560, obs_dim）
        # Bootstrapped uncertainty
        target_q_pred = []

        for i in range(self.num_qs):
            target_q_pred.append(self.target_qfs[i](obs_next.cuda(), act_next.cuda()))

        return target_q_pred

    def action_tranform(self, action, amin, amax):
        amin = torch.tensor(amin).cuda()
        amax = torch.tensor(amax).cuda()
        if self.action_n % 1 != 0:
            action = ((action - 0.5*(1+self.action_n)*amin+0.5*(self.action_n-1)*amax)/ (self.action_n*(0.000001 + amax - amin)))
        else:
            action = ((int(self.action_n) * (action - amin)) / ((0.000001 + amax - amin)))
        intger_action = (action + 0.5) // 1

        return intger_action

    def stateaction_count(self,  sa, indices, action_dim,sta):
        a = 0
        count_sa = []
        if self.action_n % 1 != 0:
            for i in range(action_dim):
                a = sa[:, i] + (self.action_n+1) * a
            a = (a + 0.5) // 1
        else:
            for i in range(action_dim):
                a = sa[:, i] + int(self.action_n+1) * a
        m = sta[indices]
        a = a.type(torch.long)
        self.num_sa[m, a] = self.num_sa[m, a] + 1
        count_sa.append(self.num_sa[m, a])

        return count_sa


    def train_from_torch(self, batch, indices,epoch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        batch_size = rewards.size()[0]
        action_dim = actions.size()[-1]

        """ Policy and Alpha Loss
		"""

        new_curr_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(obs, reparameterize=True, return_log_prob=True)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(next_obs, reparameterize=True, return_log_prob=True)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        """
		QF Loss  Q Training
		"""
	    
        q_pred = self.func(obs, actions)
        target_q_pred = self.func_target(next_obs, new_next_actions)
        q_pred_odd = self.func(obs, new_curr_actions)
        self._num_q_update_steps += 1


        a_curr = self.action_tranform(new_curr_actions, self.amin, self.amax)
        a_next = self.action_tranform(new_next_actions, self.amin, self.amax)

        sa_count  = self.stateaction_count(a_curr, indices, action_dim, self.s_curr)
        sa_next_count = self.stateaction_count(a_next, indices, action_dim, self.s_next)


        target_q_preds = torch.minimum(target_q_pred[0], target_q_pred[1]) - new_log_pi * alpha

        sa_count = torch.tensor([item.cpu().detach().numpy() for item in sa_count]).cuda()
        sa_next_count = torch.tensor([item.cpu().detach().numpy() for item in sa_next_count]).cuda()


        u_curr = self.beta * ((math.log(epoch+2, math.e)) ** (1 / 2)) / torch.sqrt(sa_count)
        u_next = self.beta * ((math.log(epoch+2, math.e)) ** (1 / 2)) / torch.sqrt(sa_next_count)
        u_curr = u_curr.transpose(0, 1)
        u_next = u_next.transpose(0, 1)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * (
            target_q_preds)
        q_target = q_target.detach()

        for qf_index in np.arange(self.num_qs):
            # Q-target
            # Critic loss. MSE. The input shape is (256,1)
            qf_loss_in = self.qf_criterion(q_pred[qf_index].float(),
                                           q_target.float())

            # For odd actions
            cat_qf_ood = torch.cat([q_pred_odd[qf_index], target_q_pred[qf_index]], 0)

            cat_qf_ood_target = torch.cat([
                torch.maximum(q_pred_odd[qf_index] - u_curr, torch.zeros(1).cuda()),
                torch.maximum(target_q_pred[qf_index] - 0.1 * u_next, torch.zeros(1).cuda())], 0)

            cat_qf_ood_target = cat_qf_ood_target.detach()

            qf_loss_ood = self.qf_criterion(cat_qf_ood.float(), cat_qf_ood_target.float())  # odd标签和真实之差

            # Final loss
            qf_loss = qf_loss_in + qf_loss_ood
            # Update the Q-functions
            self.qfs_optimizer[qf_index].zero_grad()
            qf_loss.backward(retain_graph=True)
            self.qfs_optimizer[qf_index].step()

        # Actor loss Policy Training
        q_new_actions_all = []
        for i in range(self.num_qs):
            q_new_actions_all.append(self.qfs[i](obs, new_curr_actions))
        q_new_actions = torch.min(torch.hstack(q_new_actions_all), dim=1,
                                  keepdim=True).values
        assert q_new_actions.size() == (batch_size, 1)
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()


        # Soft update the target-Nets
        for i in np.arange(self.num_qs):
            if self.prior:
                ptu.soft_update_from_to(self.qfs[i].main_network, self.target_qfs[i].main_network, self.soft_target_tau)
            else:
                ptu.soft_update_from_to(self.qfs[i], self.target_qfs[i], self.soft_target_tau)

        # Save some statistics for eval
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
			Eval should set this to None.
			This way, these statistics are only computed for one batch.
			"""
            print(sa_count)

            self.eval_statistics['Q CurrPolicy'] = np.mean(ptu.get_numpy(q_pred[0]))
            self.eval_statistics['Q NextAction'] = np.mean(ptu.get_numpy(target_q_pred[0]))

            self.eval_statistics['UCB CurrPolicy'] = np.mean(ptu.get_numpy(u_curr))
            self.eval_statistics['UCB NextAction'] = np.mean(ptu.get_numpy(u_next))
            self.eval_statistics['Q Offline'] = np.mean(ptu.get_numpy(q_pred[0]))

            self.eval_statistics['QF Loss in'] = np.mean(ptu.get_numpy(qf_loss_in))
            self.eval_statistics['QF Loss ood'] = np.mean(ptu.get_numpy(qf_loss_ood))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))

            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Targets', ptu.get_numpy(q_target)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

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
            target_qf1=self.target_qfs,
        )
