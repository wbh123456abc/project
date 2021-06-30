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
        self.env = football_env.create_environment(
                env_name="academy_3_vs_1_with_keeper", stacked=False, logdir='/tmp/football', 
                write_goal_dumps=False, write_full_episode_dumps=False, render=False, rewards='scoring,checkpoints', 
                representation='simple115')
        self.env.reset()

        # tensor for a stack of 4 frames
        self.obs_4 = np.zeros((4, 11, 11))

        # buffer to keep the maximum of last 2 frames
        self.obs_1_max = np.zeros((1, 11, 11))

        # keep track of the episode rewards
        self.rewards = []
        self.model_path = './best_trained'
        self.model = Model().to(device)
        model_state = torch.load(self.model_path)
        self.model.load_state_dict(model_state['model_params'])

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
            self.runto_9()
        else:
            episode_info = None

            # get the max of last two frames
            obs = self.obs_1_max[0]

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info

    def reset(self):
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """

        # reset OpenAI Gym environment
        self.env = football_env.create_environment(
                env_name="academy_3_vs_1_with_keeper", stacked=False, logdir='/tmp/football', 
                write_goal_dumps=False, write_full_episode_dumps=False, render=False, rewards='scoring,checkpoints', 
                representation='simple115')
        game_obs = self.env.reset()

        # reset caches
        obs = self._process_obs(game_obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []

        return self.obs_4
    def run_step(self, action):
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
            self.reset()
        else:
            episode_info = None

            # get the max of last two frames
            obs = self.obs_1_max[0]

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info

    def runto_9(self):
        while True:
            rewards = 0
            actions = 0
            done = False
            obs = np.zeros((1, 4, 11, 11))
            obs[0] = self.reset()
            for i in range(10):
                pi, v = self.model(obs_to_torch(obs))
                a = pi.sample()
                actions = a.cpu().numpy()
                obs[0], rewards, done, info = self.run_step(actions)
                if done:
                    break
            if done == False:
                break
            

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

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1)

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


class Trainer:
    """
    ## Trainer
    """

    def __init__(self, *,
                 updates: int, epochs: IntDynamicHyperParam,
                 n_workers: int, worker_steps: int, batches: int,
                 value_loss_coef: FloatDynamicHyperParam,
                 entropy_bonus_coef: FloatDynamicHyperParam,
                 clip_range: FloatDynamicHyperParam,
                 learning_rate: FloatDynamicHyperParam,
                 model_path
                 ):
        # #### Configurations

        # number of updates
        self.updates = updates
        # number of epochs to train the model with sampled data
        self.epochs = epochs
        # number of worker processes
        self.n_workers = n_workers
        # number of steps to run on each process for a single update
        self.worker_steps = worker_steps
        # number of mini batches
        self.batches = batches
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.batches
        assert (self.batch_size % self.batches == 0)

        # Value loss coefficient
        self.value_loss_coef = value_loss_coef
        # Entropy bonus coefficient
        self.entropy_bonus_coef = entropy_bonus_coef

        # Clipping range
        self.clip_range = clip_range
        # Learning rate
        self.learning_rate = learning_rate

        self.obs = np.zeros((self.n_workers, 4, 11, 11))
        # model
        self.model = Model().to(device)

        self.workers = [Worker(47)]

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

        # GAE with $\gamma = 0.99$ and $\lambda = 0.95$
        self.gae = GAE(self.n_workers, self.worker_steps, 0.99, 0.95)

        # PPO Loss
        self.ppo_loss = ClippedPPOLoss()

        # Value Loss
        self.value_loss = ClippedValueFunctionLoss()

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        ### Sample data with current policy
        """

        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 4, 11, 11), dtype=np.float32)
        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps + 1), dtype=np.float32)

        with torch.no_grad():
            # sample `worker_steps` from each worker
            for t in range(self.worker_steps):
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                '''
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w, t]))
                '''
                for w, worker in enumerate(self.workers):
                    # get results after executing the actions
                    
                    self.obs[w], rewards[w, t], done[w, t], info = worker.step(actions[w,t])

                    # collect episode info, which is available if an episode finished;
                    #  this includes total reward and length of the episode -
                    #  look at `Game` to see how it works.
                    if info:
                        tracker.add('reward', info['reward'])
                        tracker.add('length', info['length'])

            # Get value of after the final step
            _, v = self.model(obs_to_torch(self.obs))
            values[:, self.worker_steps] = v.cpu().numpy()

        # calculate advantages
        #print(rewards)
        advantages = self.gae(done, rewards, values)

        #
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values[:, :-1],
            'log_pis': log_pis,
            'advantages': advantages
        }

        # samples are currently in `[workers, time_step]` table,
        # we should flatten it for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.epochs()):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)
            
            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(mini_batch)

                # Set learning rate
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate()
                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                #print(loss)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # Update parameters based on gradients
                self.optimizer.step()
        #print(Floss)

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ### Calculate total loss
        """

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi = pi.log_prob(samples['actions'])

        # Calculate policy loss
        policy_loss = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range())

        # Calculate Entropy Bonus
        #
        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate value function loss
        value_loss = self.value_loss(value, samples['values'], sampled_return, self.clip_range())

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        loss = (policy_loss
                + self.value_loss_coef() * value_loss
                - self.entropy_bonus_coef() * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        # Add to tracker
        tracker.add({'policy_reward': -policy_loss,
                     'value_loss': value_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': self.ppo_loss.clip_fraction})

        return loss

    def run_training_loop(self, model_path):
        """
        ### Run training loop
        """
        # last 100 episode information
        model_state = {'curr_epochs': 0, 'train_steps': 0, 'model_params':dict() }
        tracker.set_queue('reward', 20, True)
        tracker.set_queue('length', 20, True)
        
        step = 0
        for update in monit.loop(self.updates):
            # sample with current policy
            samples = self.sample()
            
            # train the model
            #print('training update')
            #print(update)
            self.train(samples)

            # Save tracked indicators.
            tracker.save()
            step += 1
            # Add a new line to the screen periodically
            if step % 20 == 0:
                print(model_state['curr_epochs'])
                logger.log()
                model_state['curr_epochs'] += 1
                model_state['model_params'] = self.model.state_dict()
                torch.save(model_state, model_path)

    '''
    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))
    '''

def main():
    # Create the experiment
    experiment.create(name='ppo')
    model_path = './better_trained'
    # Configurations
    configs = {
        'updates': 1000,
        'epochs': IntDynamicHyperParam(8),
        'n_workers': 1,
        'worker_steps': 128,
        'batches': 4,
        'value_loss_coef': FloatDynamicHyperParam(0.5),
        'entropy_bonus_coef': FloatDynamicHyperParam(0.01),
        'clip_range': FloatDynamicHyperParam(0.1),
        'learning_rate': FloatDynamicHyperParam(1e-2, (0, 1e-2)),
    }

    experiment.configs(configs)

    # Initialize the trainer
    m = Trainer(
        updates=configs['updates'],
        epochs=configs['epochs'],
        n_workers=configs['n_workers'],
        worker_steps=configs['worker_steps'],
        batches=configs['batches'],
        value_loss_coef=configs['value_loss_coef'],
        entropy_bonus_coef=configs['entropy_bonus_coef'],
        clip_range=configs['clip_range'],
        learning_rate=configs['learning_rate'],
        model_path=model_path
    )

    # Run and monitor the experiment
    with experiment.start():
        m.run_training_loop(model_path)
    # Stop the workers
    #m.destroy()


# ## Run it
if __name__ == "__main__":
    main()
