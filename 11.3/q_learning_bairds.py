import random
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy


class BairdsExample:
  x = np.array([
      [2, 0, 0, 0, 0, 0, 0, 1],
      [0, 2, 0, 0, 0, 0, 0, 1],
      [0, 0, 2, 0, 0, 0, 0, 1],
      [0, 0, 0, 2, 0, 0, 0, 1],
      [0, 0, 0, 0, 2, 0, 0, 1],
      [0, 0, 0, 0, 0, 2, 0, 1],
      [0, 0, 0, 0, 0, 0, 1, 2],
  ])

  def __init__(self):
    self.state = 6

  def step(self, action):
    if action:
      self.state = 6
    else:
      self.state = random.randint(0, 5)

  def get_feature(self):
    return self.x[self.state]


class SemiGradQLearningAgent:
  def __init__(self):
    self.w = np.array([1.0, 1, 1, 1, 1, 1, 10, 1])
    self.gamma = 0.99
    self.alpha = 0.01
    self.results = []

  def step(self, env):
    action = random.random() < 1/7
    if action:
      # We do updates only if we chose a solid action.
      old_features = env.get_feature()
      env.step(action)
      new_features = env.get_feature()
      self.w += (self.alpha * self.gamma
                 * (np.dot(new_features, self.w)
                    - np.dot(old_features, self.w))
                 * old_features)
    else:
      env.step(action)
      self.results.append(deepcopy(self.w))


if __name__ == '__main__':
  NUM_STEPS = 2000
  env = BairdsExample()
  agent = SemiGradQLearningAgent()
  for i in range(NUM_STEPS):
    agent.step(env)
  plt.plot(agent.results)
  plt.xlabel('steps')
  plt.ylabel('weight value')
  plt.legend([f'w{i}' for i in range(1, 9)])
  plt.title("Q-learning on Baird's example")
  plt.savefig('q_learning_bairds.png')
