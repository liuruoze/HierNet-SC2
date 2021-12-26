import copy
import param as P


class Buffer(object):

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []

        self.values_next = []
        self.gaes = []
        self.returns = []
        self.return_values = []

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []

        self.values_next = []
        self.gaes = []
        self.returns = []
        self.return_values = []

    def append(self, obs, action, reward, value, value_next):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.values_next.append(value_next)

    def add(self, buffer):
        obs, actions, rewards, values, values_next = buffer.observations, buffer.actions, buffer.rewards, buffer.values,\
            buffer.values_next
        gaes = self.get_gaes(rewards, values, values_next)

        self.observations += obs
        self.actions += actions
        self.rewards += rewards
        return_values = self.get_return_values(buffer.rewards)

        self.values += values

        self.values_next += values_next
        self.gaes += gaes
        # self.returns += [sum(rewards)]
        self.returns += [self.get_returns(rewards)]

        self.return_values += return_values

    def get_return_values(self, rewards):
        gamma = P.gamma
        returns = copy.deepcopy(rewards)

        for t in reversed(range(len(returns) - 1)):  # is T-1, where T is time step which run policy
            returns[t] = returns[t] + gamma * returns[t + 1]
        return returns

    def get_gaes(self, rewards, v_preds, v_preds_next):
        gamma = P.gamma
        lamda = P.lamda
        deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]
        return gaes

    def get_returns(self, rewards):
        val = 0
        gamma = P.gamma
        for r in reversed(rewards):
            val = r + gamma * val

        return val


class Cnn_Buffer(object):

    def __init__(self):
        self.observations = []
        self.map_data = []
        self.battle_actions = []
        self.battle_pos = []
        self.rewards = []
        self.values = []

        self.values_next = []
        self.gaes = []
        self.returns = []

    def reset(self):
        self.observations = []
        self.map_data = []
        self.battle_actions = []
        self.battle_pos = []
        self.rewards = []
        self.values = []

        self.values_next = []
        self.gaes = []
        self.returns = []

    def append(self, obs, map_data, battle_action, battle_pos, reward, value, value_next):
        self.observations.append(obs)
        self.map_data.append(map_data)
        self.battle_actions.append(battle_action)
        self.battle_pos.append(battle_pos)
        self.rewards.append(reward)
        self.values.append(value)
        self.values_next.append(value_next)

    def add(self, buffer):

        gaes = self.get_gaes(buffer.rewards, buffer.values, buffer.values_next)

        self.observations += buffer.observations
        self.map_data += buffer.map_data
        self.battle_actions += buffer.battle_actions
        self.battle_pos += buffer.battle_pos
        self.rewards += buffer.rewards
        self.values += buffer.values

        self.values_next += buffer.values_next
        self.gaes += gaes
        # self.returns += [sum(buffer.rewards)]
        self.returns += [self.get_returns(buffer.rewards)]

    def get_gaes(self, rewards, v_preds, v_preds_next):
        gamma = P.gamma
        lamda = P.lamda
        deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]
        return gaes

    def get_returns(self, rewards):
        val = 0
        gamma = P.gamma
        for r in reversed(rewards):
            val = r + gamma * val

        return val


class Global_Buffer(object):

    def __init__(self):
        self.controller_buffer = Buffer()
        self.base_buffer = Buffer()
        self.tech_buffer = Buffer()
        self.pop_buffer = Buffer()
        self.battle_buffer = Buffer()

    def reset(self):
        self.controller_buffer.reset()
        self.base_buffer.reset()
        self.tech_buffer.reset()
        self.pop_buffer.reset()
        self.battle_buffer.reset()

    def append(self, controller, base, tech, pop, battle):
        self.controller_buffer.add(controller)
        self.base_buffer.add(base)
        self.tech_buffer.add(tech)
        self.pop_buffer.add(pop)
        self.battle_buffer.add(battle)
