import numpy as np
import pygame
import gym
from gym import spaces
import numpy as np
import tensorflow as tf         
from tensorflow import keras
import tensorflow as tf
from gym.spaces import Box
import torch
import skimage
import skimage.measure
from collections import Counter
from tensorflow.keras.layers import Dense
import time


num_agents = 5
np.random.seed(0)
ini_food_positions = np.random.randint(0,19, size=[80,2])
ini_agent_positions = np.random.randint(0,19, size=[num_agents,2])


pygame.init()
class AntColonyEnv(gym.Env):
    def __init__(self, grid_size=20, num_agents=num_agents):
        super(AntColonyEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.ant_positions = ini_agent_positions
        self.food_positions = ini_food_positions

        self.ant_labels = np.full((num_agents,1),0) 
        self.viewer = None
        self.action_space = gym.spaces.Discrete(3)  # 4 possible actions: up, down, left, right, put down, pick up
        self.observation_space = (gym.spaces.Box(low=0, high=5, shape=(num_agents, 2), dtype=np.int))

    def get_state_reward(self, ant_position, ant_label):
        x, j = ant_position
        grid = np.zeros((5,5))
        food_positions = np.array(self.food_positions)
        i = 0
        for coorx in range(x - 2, x + 3):
            j = 0
            for coory in range(j - 2, j + 3):
                if np.any(np.all(([coorx, coory]) == food_positions,axis=1)):
                    grid[i,j] = 1
                elif 20 <= coorx or coorx < 0 or 20 <= coory or coory < 0:
                    grid[i,j] = -1
                j += 1
            i += 1

        # Mark the current cell as visited
        if grid[2,2] == 1:
            c= 0
        else:
            grid[2,2] = 1
            c = -1
        b = skimage.measure.label(grid, connectivity = 2)
        ind = b[2][2]
        counts = Counter(b.flatten())
        count = counts[ind]
        count = count + c

   
        if count == 0:
            state_value = 0
        elif 0< count <= 2:
            state_value = 1
        elif 2 < count <= 4:
            state_value = 2
        elif 4 < count <= 10:
            state_value = 3
        elif 10 < count <= 25:
            state_value = 4
        state_values = [state_value, int(ant_label)] 
        return state_values, count
    
    def step(self, actions):
        rewards = []
        next_states = []
        for i in range(self.num_agents):
            action = actions[i]
            Found_food = any(np.array_equal(pos, self.ant_positions[i]) for pos in self.food_positions)
            if Found_food:
                food_label = 1
            else:
                food_label = 0

            if action == 0:  # Move up
                dirction = np.random.choice(4)
                if dirction == 0:
                    self.ant_positions[i][0] = max(0, self.ant_positions[i][0] - 1)
                elif dirction == 1:  # Move down
                    self.ant_positions[i][0] = min(self.grid_size - 1, self.ant_positions[i][0] + 1)
                elif dirction == 2:  # Move left
                    self.ant_positions[i][1] = max(0, self.ant_positions[i][1] - 1)
                elif dirction == 3:  # Move right
                    self.ant_positions[i][1] = min(self.grid_size - 1, self.ant_positions[i][1] + 1)
                rewards.append(0)
                next_state, _ = self.get_state_reward(self.ant_positions[i], self.ant_labels[i])
                next_state.append(food_label)
                next_states.append(next_state)
            elif action == 1:  # pick up
                next_state, reward = self.get_state_reward(self.ant_positions[i], self.ant_labels[i])
                if self.ant_labels[i] == 0 and Found_food:
                    self.ant_labels[i] = 1
                    rewards.append(5-reward)
                    if (type(self.food_positions) != list):
                        self.food_positions = self.food_positions.tolist()
                    new_food_positions = self.food_positions.copy()
                    for j in range(len(self.food_positions)):
                        if np.all(self.ant_positions[i] == self.food_positions[j]):
                            new_food_positions.pop(j)
                            break                       
                    self.food_positions = new_food_positions.copy()
                else:
                    rewards.append(-1)
                next_state.append(food_label)
                next_states.append(next_state)

            elif action == 2:  # put down
                next_state, reward = self.get_state_reward(self.ant_positions[i], self.ant_labels[i])
                if self.ant_labels[i] == 1 and not Found_food:
                    self.ant_labels[i] = 0
                    rewards.append(reward)
                    if (type(self.food_positions) != list):
                        self.food_positions = self.food_positions.tolist()
                    self.food_positions.append((self.ant_positions[i]).tolist())
                else:
                    rewards.append(-1)
                next_state.append(food_label)
                next_states.append(next_state)

        self.ant_positions = [self.ant_positions[i] for i in range(self.num_agents)]

        done = None
        return next_states, rewards, done, {}

       
    def reset(self):
        self.ant_positions = ini_agent_positions
        self.ant_labels = np.full((num_agents,1),0) 
        self.food_positions = ini_food_positions

        return self.ant_positions.copy()

    def render(self):
        if self.viewer is None:
            self.viewer = PygameViewer(self.grid_size)
        self.viewer.render(self.ant_positions, self.food_positions, self.ant_labels)
    

class PygameViewer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.cell_size = 10
        self.screen_size = grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size),flags=pygame.SHOWN)


    def render(self, ant_positions, food_positions, ant_labels):
        self.screen.fill((255, 255, 255))  # White background

        # Draw food
        for pos in food_positions:
            pygame.draw.rect(self.screen, (0, 255, 0), ((pos[1]) * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size))

        # Draw ants
        for i in range(len(ant_positions)):
            if ant_labels[i] == 0:
                color = (255, 0, 255)
            else:
                color = (255, 0, 0)
            pos = ant_positions[i]
            pygame.draw.rect(self.screen, color, (pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size))
        
        # pygame.event.set_grab(True)
        pygame.display.flip()
        pygame.time.delay(50)


gamma = 0.5  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off
epsilon_decay = 0.9  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum epsilon value
learning_rate = 0.001  # Learning rate for the neural network
learning_rate_decay = 0.9

class QlearningAgent:
    def __init__(self, state_size, action):
        self.state_size = state_size
        self.action = action
            
    def choose_action(self, q_values,ind):
        if np.random.rand() <= epsilon:
            action = np.random.choice(env.action_space.n)
            return action
        return np.argmax(q_values)
      
if __name__ == "__main__":
    env = AntColonyEnv()
    episodes = 100
    episode_reward = []
    state_space = [(i, j, k) for i in range(5) for j in range(2) for k in range(2)]
    state_size = 20
    Q = np.random.uniform(size = (20, 3))
    agents = []
    for i in range(num_agents):
        agent = QlearningAgent(state_size, env.action_space)
        agents.append(agent)

    for episode in range(episodes):
        env.reset()
        total_reward = 0
        # epsilon = max(episode * epsilon_decay, epsilon_min)
        # learning_rate = max(learning_rate * learning_rate_decay, 0.001)
        
        for step in range(1000):  # Adjust the maximum number of steps as needed
            if step == 0:
                states = []
                for i in range(num_agents):
                    Found_food = any(np.array_equal(pos, env.ant_positions[i]) for pos in env.food_positions)
                    if Found_food:
                        food_label = 1
                    else:
                        food_label = 0
                    state_values,_ = env.get_state_reward(env.ant_positions[i], env.ant_labels[i])
                    state_values.append(food_label)
                    states.append(state_values)

            actions = []
            current_inds = []
            for i in range(num_agents):
                current_ind = state_space.index((states[i][0],states[i][1],states[i][2]))
                action = agents[i].choose_action(Q[i], current_ind)
                actions.append(action)
                current_inds.append(current_ind)

            next_states, rewards, done, _ = env.step(actions)
            next_inds = []
            for i in range(num_agents):
                next_ind = state_space.index((next_states[i][0],next_states[i][1],next_states[i][2]))
                Q[current_ind, actions[i]] = (1 - learning_rate) * Q[current_ind, actions[i]] + \
                                                learning_rate * (rewards[i] + gamma * np.max(Q[next_ind, :]))
                next_inds.append(next_ind)

            states = next_states.copy()
            for i in range(num_agents):
                total_reward += rewards[i]
            
            # env.render()
            
            if done:
                env.reset()

                    # print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}")
            if step % 100 ==0:
                with open('Q.txt','a') as f:
                    f.write(str(Q)+'\n') 
                    f.write('Current state index ' + str(current_inds)+'\n')
                    f.write('Next state index ' + str(next_inds)+'\n')
                    f.write('Actions ' + str(actions)+'\n') 
                    f.write('Rewards ' + str(rewards)+'\n')


        print('Total Reward:', total_reward)
        episode_reward.append(total_reward)
    print(episode_reward)

    pygame.quit()



