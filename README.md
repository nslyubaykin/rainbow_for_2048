# Rainbow agent for Playing 2048 with [ReLAx](https://github.com/nslyubaykin/relax)

https://user-images.githubusercontent.com/67604207/188276886-bc7987c0-65fe-4eba-a119-7a1b7d160f64.mp4

This repository contains an [implementation](https://github.com/nslyubaykin/rainbow_for_2048/blob/master/rainbow_2048.ipynb) of 
2048 game which may be played manually in Jupyter and a custom Gym environment written on top of it. 

Then, this environment was used to train Rainbow deep q-network (Rainbow DQN) agent.

Resulting actor shows a very solid performance casually achieving 2048 tile (52% of games), and rarely (~1% of games) achieving 4096 tile.
Table of each tile frequency is shown below (100 games played):


| Tile Value  | % Games Achieved |
| ------------- | ------------- | 
| 2 | 100% |
| 4 | 100% |
| 8 | 100% |
| 16 | 100% |
| 32 | 100% | 
| 64 | 100% |
| 128 | 100% |
| 256 | 99% |
| 512 | 98% |
| 1024 | 83% |
| 2048 | 52% |
| 4096 | 1% |

Training was run for 10m environment steps.
The graph of average return vs environment step is shown below (logs done every 50k steps):

![rainbow_training](https://github.com/nslyubaykin/rainbow_for_2048/blob/master/content/training_2048.png)

The distribution of estimated Q-values vs data Q-values for rewards-clipped environment is shown below:

![rainbow_q_func](https://github.com/nslyubaykin/rainbow_for_2048/blob/master/content/q_func_2048.png)


