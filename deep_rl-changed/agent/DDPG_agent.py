#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent(BaseAgent):
    def __init__(self,action_size,env,brain,config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.action_size = action_size
        self.env = env
        self.brain = brain

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = torch.from_numpy(state).float().to(device)
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        brain_name = self.env.brain_names[0]
        if self.state is None:
            self.random_process.reset_states()

            env_info = self.env.reset(train_mode=True)[brain_name]

            self.state = env_info.vector_observations
            self.state = config.state_normalizer(self.state)
            # self.state = torch.from_numpy(self.state).float().to(device)
        if self.total_steps < config.warm_up:
            # action = [self.task.action_space.sample()]
            action = np.random.randn(1, self.action_size)
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        # action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        action = np.clip(action, -1, -1)
        #self.task.step(action)

        # next_state, reward, done, info = env.step(actions)[brain_name]

        env_info = self.env.step(action)[brain_name]           # send all actions to tne environment
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                         # get reward (for each agent)
        done = env_info.local_done                        # see if episode finished



        next_state = self.config.state_normalizer(next_state)
        # self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            # states = [torch.from_numpy(state).float().to(device) for state in states]
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
