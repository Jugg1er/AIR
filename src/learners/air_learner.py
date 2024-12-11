import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import math


class AIRLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.params = list(mac.parameters())
        self.air_params = list(mac.mlp.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.mac.alpha.requires_grad = True
        self.optimiser = Adam(params=self.params,  lr=args.lr)
        self.air_optimiser = Adam(params=self.air_params,  lr=args.lr_zeta)
        self.alpha_optimiser = Adam(params=[self.mac.alpha],  lr=args.lr_alpha)

        self.n = 0
        self.mean = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        air_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, air_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            air_out.append(air_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        air_out = th.stack(air_out, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals
        chosen_air_vals = th.gather(air_out[:, :-1], dim=4, index=th.repeat_interleave(actions.unsqueeze(3), repeats=self.n_agents, dim=3)).squeeze(4)
        chosen_air_vals = th.diagonal(chosen_air_vals, dim1=2, dim2=3)
        chosen_air_vals = th.log(chosen_air_vals)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
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
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask1 = mask.expand_as(td_error)
        mask2 = mask.expand_as(chosen_air_vals)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask1

        masked_air_vals = chosen_air_vals * mask2

        # Normal L2 loss, take mean over actual data
        air_loss = -masked_air_vals.sum() / mask2.sum()
        alpha_loss = -(self.mac.alpha * ((chosen_air_vals - self.mean) * mask2).detach()).sum() / mask2.sum()
        td_loss = (masked_td_error ** 2).sum() / mask1.sum()
        self.running_mean(chosen_air_vals.detach())

        # Optimise
        self.optimiser.zero_grad()
        td_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.air_optimiser.zero_grad()
        air_loss.backward()
        grad_norm2 = th.nn.utils.clip_grad_norm_(self.air_params, self.args.grad_norm_clip)
        self.air_optimiser.step()

        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        # print("alpha grad: ", [x.grad for x in self.alpha_optimiser.param_groups[0]['params']])
        self.alpha_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("air_loss", air_loss.item(), t_env)
            self.logger.log_stat("alpha", self.mac.alpha.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask1.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask1).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask1).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def running_mean(self, input_tensor):
        self.n += 1
        if self.n == 1:
            self.mean = input_tensor.mean()
        else:
            self.mean = self.mean * (self.n - 1) / self.n + input_tensor.mean() / self.n
    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

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
        th.save(self.air_optimiser.state_dict(), "{}/air_opt.th".format(path))
        th.save(self.alpha_optimiser.state_dict(), "{}/alpha_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.air_optimiser.load_state_dict(th.load("{}/air_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.alpha_optimiser.load_state_dict(th.load("{}/alpha_opt.th".format(path), map_location=lambda storage, loc: storage))
