import time
import wandb
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from mat.runner.shared.base_runner import Runner
from src.utils.gru_shared_buffer import GRUSharedReplayBuffer
from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from src.policy.gru_transformer_policy import GRUTransformerPolicy as Policy
import multiprocessing
import threading
import queue

def _t2n(x):
    return x.detach().cpu().numpy()

def create_policy_fn(self_config):
    return Policy(self_config['all_args'],
                  self_config['observation_space'],
                  self_config['share_observation_space'],
                  self_config['action_space'],
                  self_config['num_agents'],
                  device=self_config['device'])

def create_buffer_fn(self_config):
    return GRUSharedReplayBuffer(self_config['all_args'],
                                 self_config['num_agents'],
                                 self_config['observation_space'],
                                 self_config['share_observation_space'],
                                 self_config['action_space'],
                                 self_config['all_args'].env_name)

def create_trainer_fn(self_config, policy):
    return TrainAlgo(self_config['all_args'], policy, self_config['num_agents'], device=self_config['device'])

def multiprocess_trainer(self_config, data_queue, param_queue, log_queue):
    # Extract necessary attributes from self_config
    save_dir = self_config['save_dir']
    num_agents = self_config['num_agents']
    recurrent_N = self_config['recurrent_N']
    hidden_size = self_config['hidden_size']
    n_rollout_threads = self_config['n_rollout_threads']
    use_centralized_V = self_config['use_centralized_V']
    save_interval = self_config['save_interval']
    log_interval = self_config['log_interval']
    episode_length = self_config['episode_length']
    all_args = self_config['all_args']
    device = self_config['device']
    use_wandb = self_config['use_wandb']
    share_observation_space = self_config['share_observation_space']
    reset_env = self_config.get('reset_env', None)  # May not be needed
    # Any other attributes needed

    # Define functions within multiprocess_trainer using these variables

    def save(episode, policy):
        """Save policy's actor and critic networks."""
        policy.save(save_dir, episode)

    def log_train(train_infos, total_num_steps, buffer):
        train_infos["average_step_rewards"] = np.mean(buffer.rewards)
        # Send the log data to the log_queue
        #dtach all tensors data to avoid pickling error
        train_infos = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in train_infos.items()}

        log_queue.put((train_infos, total_num_steps))

    @torch.no_grad()
    def compute(buffer, trainer):
        """Calculate returns for the collected data."""
        trainer.prep_rollout()
        next_values = trainer.policy.get_values({key:np.concatenate(buffer.share_obs[key][-1]) for key in buffer.dict_keys},
                                                {key:np.concatenate(buffer.obs[key][-1]) for key in buffer.dict_keys},
                                                np.concatenate(buffer.rnn_states_critic[-1]),
                                                np.concatenate(buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), n_rollout_threads))
        buffer.compute_returns(next_values, trainer.value_normalizer)

    def train(trainer, buffer):
        """Train policies with data in buffer."""
        trainer.prep_training()
        train_infos = trainer.train(buffer)
        buffer.after_update()
        return train_infos

    def insert(data, buffer):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), num_agents, recurrent_N, hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), num_agents, *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), num_agents, 1), dtype=np.float32)

        if not use_centralized_V:
            share_obs = obs

        buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                      actions, action_log_probs, values, rewards, masks)

    # Start of multiprocess_trainer
    policy = create_policy_fn(self_config)
    trainer = create_trainer_fn(self_config, policy)
    buffer = create_buffer_fn(self_config)
    # No need to warmup the buffer since trainer doesn't interact with the env
    episode = 0
    while True:
        # Collect a batch of experiences from the queue
        while True:
            experience = data_queue.get()
            if experience is None:
                # Data collection is done, exit
                print("Trainer process received termination signal.")
                if buffer.step > 0:
                    compute(buffer, trainer)
                    train_infos = train(trainer, buffer)
                    param_queue.put(policy.get_state_dict())
                    print("Network parameters updated with remaining data.")
                return

            insert(experience, buffer)
            if buffer.step == 0:
                break

        # Process the batch
        if buffer.step == 0:
            compute(buffer, trainer)
            train_infos = train(trainer, buffer)
            param_queue.put(policy.get_state_dict())

            total_num_steps = (episode + 1) * episode_length * n_rollout_threads
            # save model
            if episode % save_interval == 0:
                save(episode, policy)
                print(f"Model saved at episode {episode}.")
            episode += 1
            # log information
            if episode % log_interval == 0:
                log_train(train_infos, total_num_steps, buffer)

def multiprocess_data_collection(self_config, data_queue, param_queue, episodes, episode_length):
    # Extract necessary attributes
    num_agents = self_config['num_agents']
    recurrent_N = self_config['recurrent_N']
    hidden_size = self_config['hidden_size']
    n_rollout_threads = self_config['n_rollout_threads']
    use_centralized_V = self_config['use_centralized_V']
    envs_fn = self_config['envs_fn']
    device = self_config['device']
    # Create policy and buffer
    policy = create_policy_fn(self_config)
    buffer = create_buffer_fn(self_config)
    # Create envs
    envs = envs_fn()
    # Reset envs and initialize buffer
    obs, share_obs = envs.reset()
    if not use_centralized_V:
        share_obs = obs

    if buffer.dict_keys == None:
        buffer.share_obs[0] = share_obs.copy()
        buffer.obs[0] = obs.copy()
    else:
        for key in buffer.dict_keys:
            buffer.share_obs[key][0] = share_obs[key].copy()
            buffer.obs[key][0] = obs[key].copy()

    def insert(data, buffer):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), num_agents, recurrent_N, hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), num_agents, *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), num_agents, 1), dtype=np.float32)

        if not use_centralized_V:
            share_obs = obs

        buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                      actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def collect(step, buffer, policy):
        policy.eval()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = policy.get_actions({key:np.concatenate(buffer.share_obs[key][step]) for key in buffer.dict_keys},
                                 {key:np.concatenate(buffer.obs[key][step]) for key in buffer.dict_keys},
                                 np.concatenate(buffer.rnn_states[step]),
                                 np.concatenate(buffer.rnn_states_critic[step]),
                                 np.concatenate(buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), n_rollout_threads))
        actions = np.array(np.split(_t2n(action), n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    for episode in range(episodes):
        step_total = 0
        for step in range(episode_length):
            # Check if there are new network parameters from the trainer
            try:
                while True:
                    new_params = param_queue.get_nowait()
                    policy.load_new_model(new_params)
                    print("Data Collector: Updated network parameters.")
            except queue.Empty:
                pass  # No new parameters, proceed

            values, actions, action_log_probs, rnn_states, rnn_states_critic = collect(step, buffer, policy)

            obs, share_obs, rewards, dones, infos = envs.step(actions)
            # Put the experience into the queue
            data = obs, share_obs, rewards, dones, infos, \
                   values, actions, action_log_probs, \
                   rnn_states, rnn_states_critic

            data_queue.put(data)
            insert(data, buffer)

        print(f"Data collection for Episode {episode+1}/{episodes} completed.")
    print("Data collection process completed.")
    # Signal that data collection is done
    data_queue.put(None)
    envs.close()

class GRUtopiaRunner(Runner):
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs_fn = config['envs_fn']
        self.envs = self.envs_fn(True)
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.sim_config = config["sim_config"]

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        # Create reset_env if needed
        # self.reset_env = self.envs.reset()

    def run(self):
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        data_queue = multiprocessing.Queue(maxsize=64*self.episode_length)
        param_queue = multiprocessing.Queue()
        log_queue = multiprocessing.Queue()

        # Create a configuration dictionary
        self_config = {
            'save_dir': self.save_dir,
            'num_agents': self.num_agents,
            'recurrent_N': self.recurrent_N,
            'hidden_size': self.hidden_size,
            'n_rollout_threads': self.n_rollout_threads,
            'use_centralized_V': self.use_centralized_V,
            'episode_length': self.episode_length,
            'save_interval': self.save_interval,
            'log_interval': self.log_interval,
            'all_args': self.all_args,
            'device': self.device,
            'use_wandb': self.use_wandb,
            'observation_space': self.envs.observation_space[0],
            'share_observation_space': self.share_observation_space,
            'action_space': self.envs.action_space[0],
            'envs_fn': self.envs_fn,  # Ensure this is picklable
            # Add any other necessary attributes
        }

        # Start data collector process
        collector_process = multiprocessing.Process(
            target=multiprocess_data_collection,
            args=(self_config, data_queue, param_queue, episodes, self.episode_length)
        )
        collector_process.start()

        # Start the trainer process
        trainer_process = multiprocessing.Process(
            target=multiprocess_trainer,
            args=(self_config, data_queue, param_queue, log_queue)
        )
        trainer_process.start()

        # Start logging thread
        def logging_thread(log_queue, writer):
            while True:
                log_data = log_queue.get()
                if log_data is None:
                    break
                # log_data is (train_infos, total_num_steps)
                train_infos, total_num_steps = log_data
                for k, v in train_infos.items():
                    if self.use_wandb:
                        wandb.log({k: v}, step=total_num_steps)
                    else:
                        writer.add_scalars(k, {k: v}, total_num_steps)

        log_thread = threading.Thread(target=logging_thread, args=(log_queue, self.writter))
        log_thread.start()

        # Wait for data collection and trainer processes
        collector_process.join()
        print("Data collection completed.")
        trainer_process.join()
        print("Training completed.")

        # Signal logging thread to finish
        log_queue.put(None)
        log_thread.join()
        print("Logging completed."


        # for episode in range(episodes):
        #     if self.use_linear_lr_decay:
        #         self.trainer.policy.lr_decay(episode, episodes)

        #     step_total = 0

        #     for step in range(self.episode_length):
        #         # Sample actions
        #         values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                    
        #         # Obser reward and next obs
        #         step_start = time.time()
        #         obs, share_obs, rewards, dones, infos = self.envs.step(actions)
        #         step_end = time.time()
        #         step_total += step_end - step_start

        #         data = obs, share_obs, rewards, dones, infos, \
        #                values, actions, action_log_probs, \
        #                rnn_states, rnn_states_critic 
                
        #         # insert data into buffer
        #         self.insert(data)

        #     # compute return and update network
        #     self.compute()
        #     train_start = time.time()
        #     train_infos = self.train()
        #     train_end = time.time()


        #     # post process
        #     total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
        #     # save model
        #     if (episode % self.save_interval == 0 or episode == episodes - 1):
        #         self.save(episode)

        #     # log information
        #     if episode % self.log_interval == 0:
        #         end = time.time()
                
        #         print(f"\nEpisode time: {(end-start)/self.log_interval:.2} seconds\npercent done: {episode/episodes*100:.2}% ({episode}/{episodes})")
        #         self.log_train(train_infos, total_num_steps)

        #     # eval
        #     if episode % self.eval_interval == 0 and self.use_eval:
        #         self.eval(total_num_steps)

    
    @torch.no_grad()
    def eval2(self, total_num_steps):
        for eval_env in self.eval_envs:
            eval_map = eval_env.envs[0].map_name
            eval_battles_won = 0
            eval_episode = 0

            eval_episode_rewards = []
            one_episode_rewards = []

            eval_obs, eval_share_obs, eval_available_actions = eval_env.reset()

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            while True:
                self.trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_env.step(eval_actions)
                one_episode_rewards.append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

                for eval_i in range(self.n_eval_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode += 1
                        eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                        one_episode_rewards = []
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won += 1

                if eval_episode >= self.all_args.eval_episodes:
                    eval_episode_rewards = np.array(eval_episode_rewards)
                    eval_env_infos = {eval_map + '/eval_average_episode_rewards': eval_episode_rewards}
                    self.log_env(eval_env_infos, total_num_steps)
                    eval_win_rate = eval_battles_won/eval_episode
                    print(eval_map + " eval win rate is {}.".format(eval_win_rate))
                    self.writter.add_scalars(eval_map + "/eval_win_rate", {eval_map + "/eval_win_rate": eval_win_rate},
                                             total_num_steps)
                    break

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_total_reward = {}
        eval_won = {}
        eval_episode = {}
        eval_episode_reward = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        self.trainer.prep_rollout()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_share_obs),
                                                                    np.concatenate(eval_obs),
                                                                    np.concatenate(eval_rnn_states),
                                                                    np.concatenate(eval_masks),
                                                                    np.concatenate(eval_available_actions),
                                                                    deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            eval_episode_reward += eval_rewards[:, 0, 0]
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                map_name = eval_infos[eval_i][0]['map']
                if map_name not in eval_total_reward.keys():
                    eval_total_reward[map_name] = 0
                if map_name not in eval_won.keys():
                    eval_won[map_name] = 0
                if map_name not in eval_episode.keys():
                    eval_episode[map_name] = 0

                if eval_dones_env[eval_i] and eval_episode[map_name] < self.all_args.eval_episodes:
                    eval_episode[map_name] += 1
                    eval_won[map_name] += eval_infos[eval_i][0]['won']
                    eval_total_reward[map_name] += eval_episode_reward[eval_i]
                    eval_episode_reward[eval_i] = 0

            if (np.array(list(eval_episode.values())) == self.all_args.eval_episodes).all():
                break

        for eval_map in eval_total_reward.keys():
            aver_eval_reward = eval_total_reward[eval_map] / self.all_args.eval_episodes
            self.writter.add_scalars(eval_map + "/eval_average_episode_rewards",
                                     {eval_map + "/eval_average_episode_rewards": aver_eval_reward},
                                     total_num_steps)
            aver_eval_winrate = eval_won[eval_map] / self.all_args.eval_episodes
            print(eval_map + " eval win rate is {}.".format(aver_eval_winrate))
            self.writter.add_scalars(eval_map + "/eval_win_rate", {eval_map + "/eval_win_rate": aver_eval_winrate},
                                     total_num_steps)




