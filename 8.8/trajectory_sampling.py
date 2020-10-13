import random
from collections import defaultdict
import argparse
import numpy as np
from matplotlib import pyplot as plt


class Task:
  def __init__(self, num_states, branching_factor):
    self.actions = [0, 1]
    self.states = range(num_states)  # The terminal state is excluded.
    self.start_state = 0
    self.terminal_state = num_states
    # (state, action) -> list[(prob, new_state, reward)]
    self.transition_probs = self.get_transition_probs(
        num_states, branching_factor)

  def get_transition_probs(self, num_states, branching_factor):
    transition_probs = {}
    prob = 0.9 / branching_factor
    for state in range(num_states):
      for action in self.actions:
        terminal_reward = random.gauss(0, 1)
        transition_probs[state, action] = [
            (0.1, self.terminal_state, terminal_reward)]
        possible_next_states = random.choices(
            range(num_states), k=branching_factor)
        for new_state in possible_next_states:
          reward = random.gauss(0, 1)
          transition_probs[state, action].append(
              (prob, new_state, reward))
    return transition_probs


class Estimator:
  def __init__(self, task, gamma=1):
    self.action_values = defaultdict(int)
    self.gamma = gamma
    self.results = []
    self.task = task
    # Contains the real state values ccording to the current policy.
    self.state_values = {
        state: random.randrange(-1, 1) for state in self.task.states}
    self.state_values[self.task.terminal_state] = 0

  def update(self, state, action):
    def best_action_value_from_state(state):
      return max(self.action_values[state, a] for a in self.task.actions)
    transitions = self.task.transition_probs[state, action]
    self.action_values[state, action] = sum(
        prob * (reward + self.gamma * best_action_value_from_state(new_state))
        for prob, new_state, reward in transitions)

  def evaluate_greedy_policy(self, max_delta=0.01, epsilon=0.1):
    delta = max_delta
    policy = EpsilonGreedyPolicy(
        self.action_values, self.task.actions, epsilon)
    while delta >= max_delta:
      delta = 0
      for state in self.task.states:
        old_value = self.state_values[state]
        action_probs = policy.get_probs(state)
        new_value = 0
        for action, action_prob in action_probs.items():
          for transition_prob, new_state, reward in self.task.transition_probs[state, action]:
            new_value += action_prob * transition_prob * (
                reward + self.gamma * self.state_values[new_state])
        delta = max(delta, abs(new_value - old_value))
        self.state_values[state] = new_value
    self.results.append(self.state_values[self.task.start_state])


class UniformEstimator(Estimator):
  def estimate(self, num_updates):
    updates = 0
    while True:
      for state in self.task.states:
        for action in self.task.actions:
          self.update(state, action)
          self.evaluate_greedy_policy()
          updates += 1
          if updates == num_updates:
            return self.results


class TrajectoryEstimator(Estimator):
  def __init__(self, task, gamma=1, epsilon=0.1):
    self.epsilon = epsilon
    super().__init__(task, gamma)

  def estimate(self, num_updates):
    updates = 0
    state = self.task.start_state
    while True:
      policy = EpsilonGreedyPolicy(self.action_values,
                                   self.task.actions,
                                   self.epsilon)
      action = policy.choose_action(state)
      self.update(state, action)
      self.evaluate_greedy_policy()
      updates += 1
      transitions = self.task.transition_probs[state, action]
      possible_states = [state for _, state, _ in transitions]
      possible_probs = [prob for prob, _, _ in transitions]
      state = random.choices(possible_states, possible_probs)[0]
      if state == self.task.terminal_state:
        state = self.task.start_state
      if updates == num_updates:
        return self.results


class EpsilonGreedyPolicy:
  def __init__(self, action_values, actions, epsilon):
    self.epsilon = epsilon
    self.action_values = action_values
    self.actions = actions

  def choose_action(self, state):
    if random.randrange(0, 1) < self.epsilon:
      return random.choice(self.actions)
    else:
      return self.best_action(state)

  def best_action(self, state):
    return max(self.actions, key=lambda a: self.action_values[state, a])

  def get_probs(self, state):
    action_probs = {action: self.epsilon for action in self.actions}
    action_probs[self.best_action(state)] += 1 - self.epsilon
    return action_probs


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('branching_factor', nargs=1, type=int)
  parser.add_argument('num_states', nargs=1, type=int)
  parser.add_argument('num_updates', nargs=1, type=int)
  parser.add_argument('num_runs', nargs=1, type=int)
  args = parser.parse_args()
  num_updates = args.num_updates[0]
  num_states = args.num_states[0]
  branching_factor = args.branching_factor[0]
  num_runs = args.num_runs[0]

  uniform_res = []
  trajectory_res = []
  for _ in range(num_runs):
    t = Task(num_states=num_states, branching_factor=branching_factor)
    ue = UniformEstimator(t)
    uniform_res.append(np.array(ue.estimate(num_updates)))
    te = TrajectoryEstimator(t)
    trajectory_res.append(np.array(te.estimate(num_updates)))

  plt.plot(np.array(uniform_res).mean(0))
  plt.plot(np.array(trajectory_res).mean(0))
  plt.xlabel('number of expected updates')
  plt.ylabel('value of start state under greedy policy')
  plt.legend(['uniform', 'on-policy'])
  plt.title(f'{num_states} states with b={branching_factor}')
  plt.savefig(f'num_states_{num_states}_b_{branching_factor}.png')
