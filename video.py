from typing import Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical

from labml import monit, tracker, logger, experiment
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam
from labml_helpers.module import Module
from labml_nn.rl.ppo import ClippedPPOLoss, ClippedValueFunctionLoss
from labml_nn.rl.ppo.gae import GAE

import multiprocessing
import multiprocessing.connection

import gfootball.env as football_env
import numpy as np
class Game:
    def __init__(self):
        # create environment
        self.env = None
        
        # tensor for a stack of 4 frames
        self.obs_4 = np.zeros((4, 11, 11))

        # buffer to keep the maximum of last 2 frames
        self.obs_1_max = np.zeros((1, 11, 11))

        # keep track of the episode rewards
        self.rewards = []

    def step(self, action):
        reward = 0.
        done = None

        # run for 4 steps
        for i in range(4):
            # execute the action in the OpenAI Gym environment
            obs, r, done, info = self.env.step(action)

            self.obs_1_max[0] = self._process_obs(obs)

            reward += r

            if done:
                break

        # maintain rewards for each step
        self.rewards.append(reward)

        if done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            #self.reset()
        else:
            episode_info = None

            # get the max of last two frames
            obs = self.obs_1_max[0]

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info

    def reset(self, write):
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """

        # reset OpenAI Gym environment
        self.env = football_env.create_environment(
                env_name="academy_3_vs_1_with_keeper", stacked=False, logdir='./video', 
                write_goal_dumps=False, write_full_episode_dumps=True, render=False, rewards='scoring',write_video=write,
                representation='simple115')
        obs = self.env.reset()

        # reset caches
        obs = self._process_obs(obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []

        return self.obs_4

    @staticmethod
    def _process_obs(obs):
        """
        #### Process game frames
        Convert game frames to gray and rescale to 11*11
        """
        obs = np.append(obs,np.zeros(121-115))
        res = np.resize(obs,(11,11))

        return res

class Worker:
    """
    Creates a new worker and runs it in a separate process.
    """
    def __init__(self, seed):
        self.seed = seed
        self.game = Game()

    def step(self, data):
        obs_4, reward, done, episode_info = self.game.step(data)
        return obs_4, reward, done, episode_info

    def reset(self):
        return self.game.reset()



# Select device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Model(Module):
    """
    ## Model
    """

    def __init__(self):
        super().__init__()

        # The first convolution layer takes a
        # 84x84 frame and produces a 20x20 frame
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1)

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features
        self.lin = nn.Linear(in_features=9 * 9 * 32, out_features=512)

        # A fully connected layer to get logits for $\pi$
        self.pi_logits = nn.Linear(in_features=512, out_features=19)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

        #
        self.activation = nn.ReLU()

    def __call__(self, obs: torch.Tensor):

        h = self.activation(self.conv1(obs))
        h = h.reshape((-1, 9 * 9 * 32))

        h = self.activation(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    """Scale observations from `[0, 255]` to `[0, 1]`"""
    return torch.tensor(obs, dtype=torch.float32, device=device)

def sample(model, worker_steps):
    rewards = np.zeros((1, worker_steps), dtype=np.float32)
    actions = np.zeros((1, worker_steps), dtype=np.int32)
    done = np.zeros((1, worker_steps), dtype=np.bool)
    obs = np.zeros((1, 4, 11, 11), dtype=np.float32)
    #log_pis = np.zeros((1, worker_steps), dtype=np.float32)
    #values = np.zeros((1, worker_steps + 1), dtype=np.float32)

    fuck = 0
    tot = 0
    workers = [Worker(47)]
    with torch.no_grad():
        # sample `worker_steps` from each worker
        for t in range(worker_steps):
            pi, v = model(obs_to_torch(obs))
            #values[:, t] = v.cpu().numpy()
            a = pi.sample()
            for w, worker in enumerate(workers):
                # get results after executing the actions
                
                obs[w], rewards[w, t], done[w, t], info = worker.step(actions[w,t])

                if info:
                    fuck += 1
                    print("%d Steps: %d Reward: %.2f" % (fuck, info['length'], info['reward']))
                    if info['reward'] == 1.0:
                        tot += 1
    print(tot)


def main():

    model_path1 = './best_trained'
    model_path2 = './better_trained'
    model_state1 = torch.load(model_path1)
    model_state2 = torch.load(model_path2)
    model1 = Model().to(device)
    model2 = Model().to(device)
    model1.load_state_dict(model_state1['model_params'])
    model2.load_state_dict(model_state2['model_params'])

    game = Game()
    tot = 0
    write = 0
    for t in range(1):
        write = True
        rewards = 0
        actions = 0
        done = 0
        obs = np.zeros((1, 4, 11, 11))
        obs[0] = game.reset(write)
        num = 0
        while done == 0:
            num += 1
            if num > 9:
                pi, v = model2(obs_to_torch(obs))
            else:
                pi, v = model1(obs_to_torch(obs))
            a = pi.sample()
            actions = a.cpu().numpy()
            obs[0], rewards, done, info = game.step(actions)
            if info:
                print("%d Steps: %d Reward: %.2f" % (t, info['length'], info['reward']))
                if info['reward'] == 1.0:
                    tot += 1
    
    print(tot)
    
    
        


# ## Run it
if __name__ == "__main__":
    main()
