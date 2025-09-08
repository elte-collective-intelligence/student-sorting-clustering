import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import shutil
import json


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tqdm import tqdm


class ClusterGridworld(AECEnv):
    def __init__(self, size=20, num_agents=1):
        super().__init__()
        self.size = size
        self.grid = np.zeros((self.size, self.size, 3), dtype=int)
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_vision = 2
        self.possible_agents = self.agents[:]
        self.agent_positions = {}
        self.item_types = [1, 2, 3]
        self.item_count = 10
        self.agent_carrying = {agent: None for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_carrying_rounds = {agent: 0 for agent in self.agents}
        shutil.rmtree("cluster_gridworld", ignore_errors=True)
        os.makedirs("cluster_gridworld", exist_ok=True)

    def _initialize_agents(self):
        for agent in self.agents:
            while True:
                x, y = np.random.randint(0, self.size, 2)
                if (x, y) not in self.agent_positions.values() and self.grid[
                    x, y
                ].sum() == 0:
                    self.agent_positions[agent] = (x, y)
                    self.grid[x, y, 1] = -1
                    break

    def _initialize_items(self):
        for _ in range(self.item_count):
            for item in self.item_types:
                x, y = np.random.randint(0, self.size, 2)
                while self.grid[x, y].sum() != 0:
                    x, y = np.random.randint(0, self.size, 2)
                self.grid[x, y, 0] = item

    def get_visible_items_with_proximity(self, agent):
        agent_x, agent_y = self.agent_positions[agent]
        vision_range = self.agent_vision
        visible_items = []
        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                grid_x, grid_y = agent_x + dx, agent_y + dy
                if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
                    if self.grid[grid_x, grid_y, 0] > 0:
                        distance = abs(dx) + abs(dy)
                        visible_items.append((self.grid[grid_x, grid_y, 0], distance))
        visible_items.sort(key=lambda item: item[1])
        observation = {
            "visible_items": visible_items,
        }
        return observation

    def calculate_reward_simple(
        self, agent, action, original_x, original_y, new_position, carrying, grid
    ):
        reward = 0
        x, y = new_position
        visible_items_info = self.get_visible_items_with_proximity(agent)
        reward += sum(
            1.0 / (distance + 1) for _, distance in visible_items_info["visible_items"]
        )
        if action == "pickup":
            if grid[x, y, 0] > 0 and not carrying:
                reward += 10
            else:
                reward -= 1
        elif action == "drop" and carrying:
            reward = 0
            reward += 5
        return reward

    def calculate_reward(
        self, agent, action, original_x, original_y, new_position, carrying, grid
    ):
        reward = 0
        x, y = new_position
        if action == "drop":
            if carrying is None:
                reward -= 5
            elif grid[x, y, 0] != 0:
                reward -= 5
            else:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (
                            0 <= x + dx < self.size
                            and 0 <= y + dy < self.size
                            and (dx != 0 or dy != 0)
                        ):
                            if grid[x + dx][y + dy][0] == carrying:
                                reward += 5
                                reward += 20 * sum(
                                    grid[x + dx][y + dy][0] == carrying
                                    for dx in range(-1, 2)
                                    for dy in range(-1, 2)
                                    if (dx != 0 or dy != 0)
                                    and 0 <= x + dx < self.size
                                    and 0 <= y + dy < self.size
                                )
                            elif (
                                grid[x + dx][y + dy][0] > 0
                                and grid[x + dx][y + dy][0] != carrying
                            ):
                                reward -= 10
        if action == "pickup":
            if carrying is None and grid[original_x][original_y][0] > 0:
                reward += 20
            else:
                reward -= 5
        visibility_info = self.get_visible_items_with_proximity(agent)
        proximity_reward = 0
        for item_type, distance in visibility_info["visible_items"]:
            proximity_reward += 3 / (distance + 1)
        reward += proximity_reward
        return reward

    def step(self, agent_key, action):
        x, y = self.agent_positions[agent_key]
        original_x, original_y = x, y
        if action == "up" and x > 0:
            x -= 1
        elif action == "down" and x < self.size - 1:
            x += 1
        elif action == "left" and y > 0:
            y -= 1
        elif action == "right" and y < self.size - 1:
            y += 1
        elif action == "pickup":
            if self.grid[x, y, 0] > 0 and self.agent_carrying[agent_key] is None:
                self.agent_carrying[agent_key] = self.grid[x, y, 0]
                self.grid[x, y, 0] = 0
                self.agent_carrying_rounds[agent_key] = 0
        elif action == "drop":
            if self.grid[x, y, 0] == 0 and self.agent_carrying[agent_key] is not None:
                self.grid[x, y, 0] = self.agent_carrying[agent_key]
                self.agent_carrying[agent_key] = 0
        if (x, y) != (original_x, original_y) and (
            x,
            y,
        ) not in self.agent_positions.values():
            # print(f"Moving {agent_key} from {(original_x, original_y)} to {(x, y)}")
            self.agent_positions[agent_key] = (x, y)
            self.grid[original_x, original_y, 1] = 0
            self.grid[x, y, 1] = -1

        new_position = (x, y)
        reward = self.calculate_reward_simple(
            agent_key,
            action,
            original_x,
            original_y,
            new_position,
            self.agent_carrying[agent_key],
            self.grid,
        )

        if self.agent_carrying_rounds[agent_key]:
            self.agent_carrying_rounds[agent_key] += 1

        self.rewards[agent_key] += reward
        done = self.check_if_done(agent_key)
        self.steps_taken[agent_key] += 1
        new_observation = self.observe(agent_key)
        return new_observation, reward, done

    def reset(self):
        self.grid = np.zeros((self.size, self.size, 2), dtype=int)
        self.agent_positions = {}
        self.agent_carrying = {agent: None for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.steps_taken = {agent: 0 for agent in self.agents}
        self._initialize_agents()
        self._initialize_items()
        self._agent_selector.reset()
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations

    def render(self, mode="human"):
        cmap = mcolors.ListedColormap(
            ["white", "red", "green", "blue", "yellow", "magenta", "cyan"]
        )
        norm = mcolors.BoundaryNorm([-2, 0, 1, 2, 3, 4, 5], cmap.N)
        vis_grid = self.grid[:, :, 0].copy()
        for agent, pos in self.agent_positions.items():
            x, y = pos
            if self.agent_carrying[agent] is not None:
                vis_grid[x, y] = -self.agent_carrying[agent]
            else:
                vis_grid[x, y] = -1
        plt.imshow(vis_grid, cmap=cmap, norm=norm)
        cbar = plt.colorbar(ticks=[-1.5, 0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(
            ["Agent", "Empty", "Item 1", "Item 2", "Item 3", "Agent with Item"]
        )
        for (x, y), value in np.ndenumerate(vis_grid):
            if value == 0:
                label = ""
            elif value == -1:
                label = "A"
            elif value < 0:
                label = f"A+{abs(value)}"
            else:
                label = f"I{value}"
            plt.text(y, x, label, ha="center", va="center", color="black")
        plt.xticks([]), plt.yticks([])
        if mode == "human":
            plt.show()
        else:
            plt.savefig(f"cluster_gridworld/{int(time.time()*10000)}.png")
        plt.close()

    def check_if_done(self, agent_key):
        max_steps_per_agent = 50
        return self.steps_taken[agent_key] >= max_steps_per_agent

    def observe(self, agent):
        agent_x, agent_y = self.agent_positions[agent]
        obs_range = range(-3, 4)
        observation = np.zeros((7, 7), dtype=int)
        for i, dx in enumerate(obs_range):
            for j, dy in enumerate(obs_range):
                obs_x, obs_y = agent_x + dx, agent_y + dy
                if 0 <= obs_x < self.size and 0 <= obs_y < self.size:
                    observation[i][j] = (
                        self.grid[obs_x][obs_y][0] or self.grid[obs_x][obs_y][1]
                    )
                else:
                    observation[i][j] = -2
        return observation.flatten()

    def state(self):
        return self.grid

    def seed(self, seed=None):
        np.random.seed(seed)


def create_policy_network(observation_dimensions, num_actions, hidden_sizes=(64, 64)):
    inputs = Input(shape=(observation_dimensions,))
    x = inputs
    for size in hidden_sizes:
        x = Dense(size, activation="relu")(x)
    outputs = Dense(num_actions, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_value_network(observation_dimensions, hidden_sizes=(64, 64)):
    inputs = Input(shape=(observation_dimensions,))
    x = inputs
    for size in hidden_sizes:
        x = Dense(size, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


observation_dimensions = 49
num_actions = 6

policy_network = create_policy_network(observation_dimensions, num_actions)
value_network = create_value_network(observation_dimensions)

policy_network.summary()
value_network.summary()
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def store(self, observation, action, reward, value, log_prob):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()


class MultiAgentRolloutBuffer:
    def __init__(self, agents):
        self.buffers = {agent: RolloutBuffer() for agent in agents}

    def store(self, agent, observation, action, reward, value, log_prob):
        self.buffers[agent].store(observation, action, reward, value, log_prob)

    def clear(self):
        for buffer in self.buffers.values():
            buffer.clear()

    def get(self):
        data = {
            agent: {
                "observations": np.array(buffer.observations),
                "actions": np.array(buffer.actions),
                "rewards": np.array(buffer.rewards),
                "values": np.array(buffer.values),
                "log_probs": np.array(buffer.log_probs),
            }
            for agent, buffer in self.buffers.items()
        }
        return data


actions_str = ["up", "down", "left", "right", "pickup", "drop"]


def collect_data(env, policy_network, value_network, buffer, steps_per_episode):
    observations = env.reset()
    for _ in range(steps_per_episode):
        for agent_id in env.agents:
            observation = observations[agent_id]
            observation = np.expand_dims(observation, axis=0)

            logits = policy_network.predict(observation, verbose=0)
            action_prob = tf.nn.softmax(logits).numpy()
            action = np.random.choice(len(action_prob[0]), p=action_prob[0])
            log_prob = tf.math.log(action_prob[0][action])
            value = value_network.predict(observation, verbose=0)

            action_str = actions_str[action]
            new_observation, reward, done = env.step(agent_id, action_str)
            # env.render("non")
            observations[agent_id] = new_observation if not done else None

            buffer.store(
                agent_id,
                observation.squeeze(),
                action,
                reward,
                value.squeeze(),
                log_prob,
            )
            if done:
                break


def compute_gae(rewards, values, gamma, lam, next_value):
    values = values + [next_value]
    gae = 0

    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages


def compute_ppo_loss_and_gradients(
    policy_model,
    value_model,
    observations,
    actions,
    advantages,
    returns,
    old_log_probs,
    clip_ratio,
):
    with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
        new_logits = policy_model(observations, training=True)
        new_probs = tf.nn.softmax(new_logits)
        actions_one_hot = tf.one_hot(actions, depth=num_actions)

        new_log_probs = tf.math.log(tf.reduce_sum(new_probs * actions_one_hot, axis=1))

        ratios = tf.exp(new_log_probs - old_log_probs)

        clipped_ratios = tf.clip_by_value(ratios, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratios * advantages, clipped_ratios * advantages)
        )

        values = tf.squeeze(value_model(observations, training=True))
        value_loss = tf.reduce_mean(tf.square(returns - values))

        total_loss = policy_loss + 0.5 * value_loss

    policy_grads = policy_tape.gradient(total_loss, policy_model.trainable_variables)
    value_grads = value_tape.gradient(total_loss, value_model.trainable_variables)

    return total_loss, policy_grads, value_grads


num_agents = 2
observation_dimensions = 5 * 5
num_actions = 6
gamma = 0.99
lam = 0.97
clip_ratio = 0.2
policy_learning_rate = 1e-2
value_function_learning_rate = 5e-3
num_epochs = 100
steps_per_episode = 50

# Initialize environment and buffer
env = ClusterGridworld(size=20, num_agents=num_agents)
buffer = MultiAgentRolloutBuffer(env.agents)
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_function_learning_rate)
losses = []
rewards = []
for epoch in tqdm(range(num_epochs)):
    collect_data(env, policy_network, value_network, buffer, steps_per_episode)

    data = buffer.get()

    all_observations, all_actions, all_rewards, all_values, all_log_probs = (
        [],
        [],
        [],
        [],
        [],
    )

    for agent_id, agent_data in data.items():
        all_observations.extend(agent_data["observations"])
        all_actions.extend(agent_data["actions"])
        all_rewards.extend(agent_data["rewards"])
        all_values.extend(agent_data["values"])
        all_log_probs.extend(agent_data["log_probs"])
    next_value = 0
    returns, advantages = compute_gae(all_rewards, all_values, gamma, lam, next_value)

    total_loss, policy_grads, value_grads = compute_ppo_loss_and_gradients(
        policy_network,
        value_network,
        np.array(all_observations),
        np.array(all_actions),
        np.array(advantages),
        np.array(returns),
        np.array(all_log_probs),
        clip_ratio,
    )

    policy_optimizer.apply_gradients(
        zip(policy_grads, policy_network.trainable_variables)
    )
    value_optimizer.apply_gradients(zip(value_grads, value_network.trainable_variables))

    buffer.clear()

    print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.numpy():.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {np.mean(all_rewards):.4f}")
    losses.append(total_loss.numpy())
    rewards.append(np.mean(all_rewards))

print(losses)
print(rewards)


def evaluate_agent(env, policy_network, num_episodes=300):
    total_rewards = 0.0
    observation = env.reset()

    for _ in range(num_episodes):
        done = {agent: False for agent in env.agents}
        episode_reward = 0.0

        for agent_id in env.agents:
            observation = env.observe(agent_id)
            observation = np.expand_dims(observation, axis=0)
            action_logits = policy_network.predict(observation)

            action = np.argmax(
                action_logits[0]
            )  # Choose action with highest probability
            action_str = actions_str[action]
            observation, reward, done[agent_id] = env.step(agent_id, action_str)
            episode_reward += reward

            env.render("non")

        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    return average_reward


average_reward = evaluate_agent(env, policy_network)
print(average_reward)
