import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


class HGCNLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.graph_update_interval=args.graph_update_interval
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                assert args.common_reward, "VDN only supports common reward setting"
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                assert args.common_reward, "QMIX only supports common reward setting"
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        # Calculate estimated Q-Values
        mac_out = []
        hyper_graphs = []  # 用于收集各个时间步的超图
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            print(f"t in hgcn_learner range line 77:{t}")
            agent_outs,current_hg  = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            hyper_graphs.append(current_hg)  # [batch_size, n_agents, n_hyper_edges]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim
        

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs,_ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error**2).sum() / mask.sum()

        #计算 skewness_loss - 每隔 args.graph_update_interval 步才计算一次
        skewness_loss_sum = 0.0
        calc_count = 0

        # 遍历时间步
        for t in range(batch.max_seq_length):
            # 只在满足 graph_update_interval 的时刻计算 skewness
            if t % self.args.graph_update_interval == 0:
                H_t = hyper_graphs[t]  # [batch_size, n_agents, n_hyper_edges]

                # 计算超边大小 S_j = sum_i H_{ij}
                S_t = th.sum(H_t, dim=1)  # [batch_size, n_hyper_edges]
                M = S_t.size(1)

                mu_t = th.mean(S_t, dim=1, keepdim=True)
                sigma_sq_t = th.mean((S_t - mu_t) ** 2, dim=1)
                sigma_t = th.sqrt(sigma_sq_t + 1e-8)

                third_moment_t = th.mean((S_t - mu_t) ** 3, dim=1)
                skewness_t = third_moment_t / (sigma_t ** 3 + 1e-8)  # [batch_size]

                # Sigmoid -> [-1, 1]
                skewness_sig_t = 2.0 * th.sigmoid(skewness_t) - 1.0

                # 与目标做差
                target_skew = self.args.target_skewness
                skewness_loss_t = th.mean((skewness_sig_t - target_skew) ** 2)

                skewness_loss_sum += skewness_loss_t
                calc_count += 1

        # 若本条序列内没有任何时间步满足 graph_update_interval，也可以做个保护
        if calc_count > 0:
            skewness_loss_final = skewness_loss_sum / calc_count
        else:
            skewness_loss_final = 0.0

        total_loss = td_loss + skewness_loss_final
        if th.isnan(td_loss).any():
            print("NaN detected in loss computation.")
            import pdb; pdb.set_trace()
        if th.isnan(skewness_loss_final).any():
            print("NaN detected in loss computation.")
            import pdb; pdb.set_trace()
        # Optimise
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("skewness_loss_final", skewness_loss_final.item(), t_env)
            self.logger.log_stat("total_loss", total_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
