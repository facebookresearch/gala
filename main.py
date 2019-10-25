# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import glob
import pprint as pp
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp

from gala.arguments import get_args
from gala.storage import RolloutStorage
from gala.model import Policy
from gala.gpu_gossip_buffer import GossipBuffer
from gala.gala_a2c import GALA_A2C
from gala.graph_manager import FullyConnectedGraph as Graph


def actor_learner(args, rank, barrier, device, gossip_buffer):
    """ Single Actor-Learner Process """

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.set_device(device)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # (Hack) Import here to ensure OpenAI-gym envs only run on the CPUs
    # corresponding to the processes' affinity
    from gala import utils
    from gala.envs import make_vec_envs
    # Make envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_procs_per_learner,
                         args.gamma, args.log_dir, device, False,
                         rank=rank)

    # Initialize actor_critic
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    # Initialize agent
    agent = GALA_A2C(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm,
        rank=rank,
        gossip_buffer=gossip_buffer
    )

    rollouts = RolloutStorage(args.num_steps_per_update,
                              args.num_procs_per_learner,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    # Synchronize agents before starting training
    barrier.wait()
    print('%s: barrier passed' % rank)

    # Start training
    start = time.time()
    num_updates = int(args.num_env_steps) // (
        args.num_steps_per_update
        * args.num_procs_per_learner
        * args.num_learners)
    save_interval = int(args.save_interval) // (
        args.num_steps_per_update
        * args.num_procs_per_learner
        * args.num_learners)

    for j in range(num_updates):

        # Decrease learning rate linearly
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)
        # --/

        # Step through environment
        # --
        for step in range(args.num_steps_per_update):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        # --/

        # Update parameters
        # --
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()
        # --/

        # Save every "save_interval" local environment steps (or last update)
        if (j % save_interval == 0
                or j == num_updates - 1) and args.save_dir != '':
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(args.save_dir,
                            '%s.%.3d.pt' % (rank, j // save_interval)))
        # --/

        # Log every "log_interval" local environment steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            num_steps = (j + 1) * args.num_procs_per_learner \
                * args.num_steps_per_update
            end = time.time()
            print(('{}: Updates {}, num timesteps {}, FPS {} ' +
                   '\n {}: Last {} training episodes: ' +
                   'mean/median reward {:.1f}/{:.1f}, ' +
                   'min/max reward {:.1f}/{:.1f}\n').
                  format(rank, j, num_steps,
                         int(num_steps / (end - start)), rank,
                         len(episode_rewards),
                         np.mean(episode_rewards),
                         np.median(episode_rewards),
                         np.min(episode_rewards),
                         np.max(episode_rewards),
                         dist_entropy, value_loss, action_loss
                         ))
        # --/


def make_gossip_buffer(args, mng, device):

    # Make local-gossip-buffer
    if args.num_learners > 1:
        # Make Topology
        topology = []
        for rank in range(args.num_learners):
            graph = Graph(rank, args.num_learners,
                          peers_per_itr=args.num_peers)
            topology.append(graph)

        # Initialize "actor_critic-shaped" parameter-buffer
        actor_critic = Policy(
            (4, 84, 84),
            base_kwargs={'recurrent': args.recurrent_policy},
            env_name=args.env_name)
        actor_critic.to(device)

        # Keep track of local iterations since learner's last sync
        sync_list = mng.list([0 for _ in range(args.num_learners)])
        # Used to ensure proc-safe access to agents' message-buffers
        buffer_locks = mng.list([mng.Lock() for _ in range(args.num_learners)])
        # Used to signal between processes that message was read
        read_events = mng.list([
            mng.list([mng.Event() for _ in range(args.num_learners)])
            for _ in range(args.num_learners)])
        # Used to signal between processes that message was written
        write_events = mng.list([
            mng.list([mng.Event() for _ in range(args.num_learners)])
            for _ in range(args.num_learners)])

        # Need to maintain a reference to all objects in main processes
        _references = [topology, actor_critic, buffer_locks,
                       read_events, write_events, sync_list]
        gossip_buffer = GossipBuffer(topology, actor_critic, buffer_locks,
                                     read_events, write_events, sync_list,
                                     sync_freq=args.sync_freq)
    else:
        _references = None
        gossip_buffer = None

    return gossip_buffer, _references


def train(args):
    pp.pprint(args)

    proc_manager = mp.Manager()
    barrier = proc_manager.Barrier(args.num_learners)

    # Shared-gossip-buffer on GPU-0
    device = torch.device('cuda:%s' % 0 if args.cuda else 'cpu')
    shared_gossip_buffer, _references = make_gossip_buffer(
        args, proc_manager, device)

    # Make actor-learner processes
    proc_list = []
    for rank in range(args.num_learners):

        # Uncomment these lines to use 2 GPUs
        # gpu_id = int(rank % 2)  # Even-rank agents on gpu-0, odd-rank on gpu-1
        # device = torch.device('cuda:%s' % gpu_id if args.cuda else 'cpu')
        proc = mp.Process(
            target=actor_learner,
            args=(args, rank, barrier, device, shared_gossip_buffer),
            daemon=False
        )
        proc.start()
        proc_list.append(proc)

        # # Bind agents to specific hardware-threads (generally not necessary)
        # avail = list(os.sched_getaffinity(proc.pid))  # available-hwthrds
        # cpal = math.ceil(len(avail) / args.num_learners)  # cores-per-proc
        # mask = [avail[(rank * cpal + i) % len(avail)] for i in range(cpal)]
        # print('process-mask:', mask)
        # os.sched_setaffinity(proc.pid, mask)

    for proc in proc_list:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    args = get_args()
    torch.set_num_threads(1)

    # Make/clean save & log directories
    # --
    def remove_files(files):
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
    try:
        os.makedirs(args.log_dir)
    except OSError as e:
        print(e)
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        remove_files(files)
    try:
        os.makedirs(args.save_dir)
    except OSError as e:
        print(e)
        files = glob.glob(os.path.join(args.save_dir, '*.pt'))
        remove_files(files)
    # --/

    train(args)
