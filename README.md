# Clustering-TorchRL Project

## Overview
This project investigates a multi-agent reinforcement learning (MARL) problem implemented using PyTorch and TorchRL. The goal is to study cooperative behavior and clustering dynamics in a grid-based environment using Proximal Policy Optimization (PPO). The project emphasizes reproducibility, modular configuration, and collaborative development.

---

## Project Structure (Benedek)
The project structure was implemented to provide the necessary means of organisation for an RL pipeline. <br>

.
├── configs/
│ ├── env/
│ ├── algo/
│ ├── agent/
│ └── experiment/
├── docker/
│ ├── Dockerfile
│ └── entrypoint.sh
├── src/
│ ├── envs/
│ ├── agents/
│ ├── utils/
│ ├── main.py
│ └── run.py
├── test/
├── requirements.txt
├── requirements.lock.txt
└── README.md

yaml
Copy code

---

## Environment (Benedek)

The environment went through multiple iterations. The first instinct was that this task needs five distinct actions. Four for movement and one for picking up or putting down an object. This is the only part of the environment that did not went through any iterations. The other main design choice was that the environment should separate the components into different channels. This was implemented so that the convolutional neural network could learn different things regarding walls candy and agents. The first version of the environment that can be seen in *version_1* and *simulation_version_one.gif* used a single channel for everything. This means that there was 3 channels, one for walls, one for candy and one for agents. This made the training very challenging as agents needed an other field with their position and candies were very hard to differentiate since they could only be denoted with numbers. I considered a cathegorical encoding vector, however that did not fit into the idea of using a single channel for the candies. <br>
Following this the environment went through multiple iterations, that I will not detail as the final version contains most of the improvements. <br>
<b>Pheromone environment (version 5)<b> This environment uses one channel for walls one for agents, and 2 for every candy. One is the actual locations of the candies of a given colour and the other is a pheromone layer emitted by these candies. This is important for multiple reasons. If we use differen channels for different candies, the network does not need to learn to differentiate between them numerically as they are already on different channels. Additionally, since the agents have limited vision, this pheromone filed helps to guide them towards candies in the exploration phase. This was needed as there were instances of agents being stuck because they did not see any candy. This resulted in the reward for moving and the reward for staying in place being equal.<br> 
Pheromone propagations was an other challange that was at the end implemented with a two dimensional fixed, normalised and mean centred kernel, that was run in every step. The construction uses some guardrails as the pheromone has a decay rate so that it cannot propagate indefinitely making the whole map uniformly one. This provides a highway for agents on which they can find candy.<br>
Adding these made convergance much faster (albeit, still not feasably fast). The final reward function is the following: Penalise for taking a step (if the agent does not move it only collects this negative penalty), penalise for picking up a candy from a cluster, reward for cleaning up a lonly candy, reward for moving towards a cluster with a candy in hand and reward for placing a candy in a cluster.

---

## Learning Algorithm (Benedek)

The project usese a CTDE approach with clipped PPO. <br>
The models for the different versions can be found in *src/models*. Even in the first version a convolutional neural network can be found, however that was already the second iteration of the networks. I have experimented with linear models, however due to the complexity of the problem they were unable to grasp the spatial structure, therfor the introduction of CNNs. 
<br><b>The actor module<b> has a truncated view of the whole environment. In the largest model trained it is a 5 by 5 view of the environmnet. The means for this are implemented in the Pheromone environment. (src/version_9) <b>The critique model<b> takes the whole state as the input, however it is not used during the evaluation of the model. (such as: src/version_9/visualise_debug visualise_trained and gif_creation)<br>
<b>Disclosure<b>: it is important to mention here that between the versions an improvement in finding and picking a candy can be observed, however proper clustering behaviour is not reached due to the lack of computing power that was at my disposal. The highest number of steps for which I could push was 200_000 which shows promising results. 

---
## Visualisation tools (Benedek)

An important step for debugging and playing out the simulation is visualisation, therefore the multitude of the visualisation scripts. In the first version pygame and matplotlib was prefered, whereas for the last version I have transitioned to *curses* as it is much more lightweight and handles the same task. However the gif creation still uses matplotlib. 

---

## Configuration Management (Arron)

Configuration management in this project is handled using Hydra, enabling a structured, modular, and reproducible approach to experiment design. All parameters relevant to the environment, learning algorithm, agent architecture, and experimental setup are decoupled from the core codebase and stored in hierarchical configuration files.

Hydra is used to manage:

Environment parameters, such as grid size, number of agents, observation radius, and reward-related settings.

Algorithm hyperparameters, including learning rate, PPO clipping coefficient, entropy regularization strength, batch sizes, number of optimization epochs, and discount factors.

Agent settings, such as policy-sharing behavior, actor–critic network configurations, and execution device (CPU or GPU).

Experiment-level parameters, including random seeds and run identifiers to ensure deterministic and repeatable experiments.

This configuration-driven design allows experiments to be reproduced exactly by specifying the same configuration files or command-line overrides. It also enables rapid experimentation through parameter sweeps without modifying source code, supporting systematic evaluation of different training regimes and architectural choices.

## Docker and Reproducibility (Arron)

To ensure full reproducibility and platform independence, the entire project is containerized using Docker. The Docker setup encapsulates the operating system, Python runtime, and all required dependencies, eliminating discrepancies between development environments.

The Docker image is built using a version-pinned dependency specification, ensuring that:

All experiments run with the same library versions.

Results are consistent across different machines and operating systems.

The project can be executed without manual environment setup.

The image is built using the following command:

docker build -t clustering-torchrl -f docker/Dockerfile .


Once built, the container provides a single-entry execution point for training or evaluation. This setup guarantees that experiments can be reproduced at any time by rebuilding the image and running the container, satisfying the reproducibility requirements of the assignment.

## Reports, Testing, and Parameter Exploration (Arron)

Extensive testing and experimental validation were conducted throughout development to ensure correctness, stability, and robustness of the system. This included both formal tests and empirical experimentation.

Testing

The project includes smoke tests and sanity checks that verify:

Correct initialization, reset, and stepping of the multi-agent environment.

Compatibility of environment outputs with TorchRL collectors and loss modules.

Stability of the training loop across full training runs without runtime errors.

These tests help catch regressions early and ensure that changes in configuration or architecture do not break core functionality.

## Reporting and Analysis 

Training progress is monitored through logged metrics such as reward trends and loss values. These metrics provide quantitative evidence of learning stability and convergence behavior. In addition, qualitative analysis is supported through visualization outputs that illustrate agent behavior and clustering dynamics over time.

---

## Run training:

docker run --rm clustering-torchrl train


## Run evaluation:

docker run --rm clustering-torchrl eval


## All dependencies are version-pinned in requirements.lock.txt.

Testing

Basic tests are included to ensure correctness and reproducibility.

Run tests:

pytest -q
