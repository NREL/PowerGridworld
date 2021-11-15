Examples of training MARL policies using RLLib.  Do

```
pip install -r requirements.txt
```

to intall the required dependencies in your environment.

__Note:__  RLLib is a sophisticated software package, and there are many 
hyperparameters involved in running any given algorithm 
(see, e.g., [here](https://docs.ray.io/en/latest/rllib-algorithms.html#ppo)).  
Experiments for our paper were run using compute nodes on NREL's Eagle 
supercomputer, with 34 CPU rollout workers and 1 GPU learner.  
(Multi-node jobs are also possible but are architecture-dependent, 
we have Slurm-based scripts that we can share on request).  Most 
users running on a local machine won't have access to these types of resources 
which may affect how you want to run the training.  Some considerations:

1.  The `train_batch_size` parameter denotes the total number of environment steps
used for each policy update.  If this number is large (e.g., 10k) but the `num_workers` 
is small (e.g., 4), it may take a very long time for the workers to collect each batch.
In our example, this would result in each worker needing to collect 2500 steps or
10 complete episodes.  Consider using a smaller `train_batch_size` in this case.

2. The `rollout_fragment_length` hyperparameter is RLLib's way of letting you decide
how to break up episodes before sending trajectories to the learner.  We set this 
value equal to the episode length so that there weren't "boundaries" in the training data, 
but this is not strictly necessary. A related parameter is the `batch_mode` which 
determines whether the agent will allow episodes to be cut short (`truncate_episodes`) 
or will require that the episode finish before performing policy updates 
(`complete_episodes`).  Using the defaults we provide 
(`rollout_fragment_length = env.max_episode_steps` and 
`batch_mode = complete_episodes`) should work for local training, so long as the
`train_batch_size` is sufficiently small (see previous bullet).

3. Finally, training batch size -- the amount of data the agent uses for policy
updates -- turns out to be a significant parameter in how/if the algorithm 
converges.  You may have to tune other parameters such as the learning rate (`lr`)
which are known to be tightly correlated with batch size in terms of learning 
performance.  See, e.g., https://arxiv.org/abs/1812.06162 for more discussion
and references.  You can use [Ray's Tune library](https://docs.ray.io/en/latest/tune/index.html) 
to help with hyperparameter tuning.

Happy learning!
