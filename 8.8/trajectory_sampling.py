import random
from collections import defaultdict


class Task:
  def __init__(self, num_states, branching_factor):
    self.actions = [0, 1]
    self.states = range(num_states)  # The terminal state is excluded.
    # (state, action) -> list[(prob, new_state, reward)]
    self.transition_probs = self.get_transition_probs(
        num_states, branching_factor)

  def get_transition_probs(self, num_states, branching_factor):
    transition_probs = {}
    prob = 0.9 / branching_factor
    for state in range(num_states):
      for action in self.actions:
        terminal_state = num_states + 1
        terminal_reward = random.gauss(0, 1)
        transition_probs[state, action] = [
            (0.1, terminal_state, terminal_reward)]
        possible_next_states = random.choices(
            range(num_states), k=branching_factor)
        for new_state in possible_next_states:
          reward = random.gauss(0, 1)
          transition_probs[state, action].append(
              (prob, new_state, reward))
    return transition_probs


class Estimator:
  def __init__(self, task, gamma=1):
    self.acion_values = defaultdict(int)
    self.gamma = gamma
    self.start_state = 0
    self.results = []
    self.task = task
    # Contains the real state values ccording to the current policy.
    self.state_values = {state: 0 for state in self.task.states}

  def update(self, state, action):
    def best_action_value_from_state(state):
      return max(self.action_values[state, a] for a in self.actions)
    transitions = self.task.transition_probs[state, action]
    self.action_values[state, action] = sum(
        prob * (reward + self.gamma * best_action_value_from_state(new_state)
                for prob, new_state, reward in transitions))

  def evaluate_greedy_policy(self, max_delta, epsilon):
    delta = max_delta
    policy = EpsilonGreedyPolicy(
        self.action_values, self.task.actions, epsilon)
    while delta > max_delta:
      delta = 0
      for state in self.task.states:
        old_value = self.state_values[state]
        action_probs = policy.get_probs(state)
        new_value = 0
        for action, action_prob in action_probs:
          for transition_prob, new_state, reward in self.task.transition_probs[state, action]:
            new_value += action_prob * transition_prob * (
                reward + self.gamma * self.state_values[new_state])
        delta = max(delta, math.abs(new_value - old_value))
        self.state_values[state] = new_value
    self.result.append(self.state_values[self.start_state])


class UniformEstimator(Estimator):
  def estimate(self, num_updates):
    updates = 0
    while True:
      for state, action in zip(range(self.task.num_states), self.task.actions):
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
    while True:
      state = self.start_state
      policy = EpsilonGreedyPolicy(self.action_values,
                                   self.task.actions,
                                   self.epsilon)
      action = policy.choose_action(state)
      self.update(state, action)
      self.evaluate_greedy_policy()
      updates += 1
      if updates == num_updates:
        return self.results


class EpsilonGreedyPolicy:
  def __init__(self, action_values, actions, epsilon):
    self.epsilon = epsilon
    self.action_values = action_values
    self.actions = actions

  def choose_action(state):
    if random.random(0, 1) < self.epsilon:
      return random.choice(actions)
    else:
      return max(actions, key=lambda a: self.action_values[state, a])


if __name__ == '__main__':
  t = Task(num_states=1000, branching_factor=1)
  ue = UniformEstimator(t)
  te = TrajectoryEstimator(t)
