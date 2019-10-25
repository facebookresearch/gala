# Gossip-based Actor-Learner Architectures (GALA)
This repo contains the implementation of GALA used for the experiments reported in
> Mido Assran, Joshua Romoff, Nicolas Ballas, Joelle Pineau, and Mike Rabbat, "Gossip-based actor learner architectures for deep reinforcement learning," *Advances in Neural Information Processing Systems (NeurIPS)* 2019. [arxiv version](https://arxiv.org/abs/1906.04585)

## Environment Setup
This code has been tested with
* Python 3.7.4
* PyTorch 1.0 or higher
The experiments reported in the paper were run using PyTorch 1.0. We have also tested this code with PyTorch 1.3.

### Install and Modify OpenAI Baselines
We use a modified version of the OpenAI Baselines interface to run our experiments. The modifications make it possible to efficiently run multiple environment instances in parallel (on a server with multiple CPUs) using Python's `multiprocessing` library.
```
# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```
After installing the latest version of baselines, open the file ```baselines/common/vec_env/shmem_vec_env.py```, go to the definition of  ```ShmemVecEnv.__init__(...)``` and change the default value of `context` from `spawn` to ```fork```.


### Other requirements
To install other requirements, return to the GALA repo directory and run 
```
pip install -r requirements.txt
```

### Running the code
As an example, to use GALA-A2C to train an agent to play the `PongNoFrameskip-v4` environment using 4 actor-learners and 16 simulators per actor-learner, run
```
OMP_NUM_THREADS=1 python -u main.py --env-name 'PongNoFrameskip-v4' \
    --user-name $USER --seed 1 --lr 0.0014 \
    --num-env-steps 40000000 \
    --save-interval 500000 \
    --num-learners 4 \
    --num-peers 1 \
    --sync-freq 100000000 \
    --num-procs-per-learner 16 \
    --save-dir '/gala_test/models/Pong/' \
    --log-dir '/gala_test/logs/Pong/'
```

This code produces one log file for each simulator. The log file contains three columns, the reward, episode length, and wall clock time, recorded after every episode.

## Acknowledgements
This code is based on [Ilya Kostrikov](https://github.com/ikostrikov)'s [pytorch-a2c-ppo-acktr-gail repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

We're also grateful to the authors of [torchbeast](https://github.com/facebookresearch/torchbeast). We used a pre-release version to obtain the comparison with [Impala](https://arxiv.org/abs/1802.01561) reported in the paper.

## License
See the LICENSE file for details about the license under which this code is made available.

