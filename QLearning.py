import numpy as np
import random


class QAgent:
    def __init__(self, width=3, height=2, all_states=None, rewards=None, mines_number=0, start_state=(0, 0),
                 end_state=(0, 3), actions=None, landmines=None, Q_values=None):
        self.width = width
        self.height = height
        self.gamma = 0.75

        self.start_state = start_state
        self.end_state = end_state
        self.mines_number = mines_number
        self.record = []
        self.reshaped_record = []
        self.iterations = 0
        self.opt_policy = []

        # Define all states
        if all_states is None:
            all_states = []
            for i in range(height):
                for j in range(width):
                    all_states.append((i, j))
        self.all_states = all_states

        # Dictionary of possible actions.
        if actions is None:
            actions = ('DOWN', 'UP', 'LEFT', 'RIGHT')
            self.actions = actions

        # Setup landmines
        self.landmines = []
        # delete start and end states from a our list of states
        temp = self.all_states.copy()

        for state in all_states:
            # remove state if equal to start or end states
            if state == start_state:
                temp.remove(state)
            if state == end_state:
                temp.remove(state)

        # Choose random states for landmines
        if landmines is None:
            self.landmines = random.sample(temp, mines_number)

        # Define rewards dictionary for all states
        if rewards is None:
            rewards = {}

            # setup rewards for all states
            for i in self.all_states:
                # setup start state reward
                if i == start_state:
                    rewards[i] = 0
                # setup end state reward
                elif i == end_state:
                    rewards[i] = 1000
                else:
                    rewards[i] = 0

            # setup rewards for landmines
            for i in self.landmines:
                rewards[i] = -500

        self.rewards = rewards

        # Define an initial policy
        self.policy = {}
        for state in self.all_states:
            # select random policy
            self.policy[state] = np.random.choice(self.actions)

        # Define initial Q-Value Function
        Q_values = {}

        for state in self.all_states:
            # Set values in all to be 0
            Q_values[state] = [0, 0, 0, 0]
        self.Q_values = Q_values

        for state in self.all_states:
            self.record.append(0)

        print(f'Q_Values: {self.Q_values}')
        print(f'Rewards: {self.rewards}')
        print(f'Landmines: {self.landmines}')
        print(f'Policy: {self.policy}')
        print(f'Actions: {self.actions}')
    # .....................................................................................................end of Class

    def algorithm(self):

        discount_factor = 0.9
        epsilon = 0.9
        learning_rate = 0.9

        # Pick up a state randomly for episode
        current_state = self.get_random_starting_location()

        for episode in range(1000):
            # until we reach terminal state:
            while not self.is_terminal_state(current_state):
                move_index = self.get_next_move(current_state, epsilon)

                # perform action
                # store old state
                old_state = current_state
                current_state = self.get_next_location(current_state, move_index)
                # print(f'current_state: {current_state}')

                # obtain the reward for moving to the new state,
                reward = self.rewards[current_state]
                old_Q_value = self.Q_values[old_state][move_index]

                # calculate the temporal difference
                temporal_difference = reward + (discount_factor * np.max(self.Q_values[current_state])) - old_Q_value

                # update the Q values for prev Q(s,a) pairs
                new_Q_value = old_Q_value + (learning_rate * temporal_difference)
                self.Q_values[old_state][move_index] = new_Q_value

        print('Training complete')
        print(f'After Q_values: {self.Q_values}')

    # define a function that determines if the specified location is a terminal state
    def is_terminal_state(self, state):
        # if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
        if state == self.end_state:
            return True
        else:
            return False

    # define a function that will choose a random, non-terminal starting location
    def get_random_starting_location(self):
        # get a random state
        random_state = random.choice(self.all_states)

        # repeat if state is the terminal state,until a terminal is defined
        while self.is_terminal_state(random_state):
            random_state = random.choice(self.all_states)

        return random_state

    # define a greedy algorithm that will choose which action to take next
    def get_next_move(self, current_state, epsilon):
        # ifa  chosen value between 0 and 1 is less than epsilon,
        # then choose the most promising value from the Q-table for this state.
        if np.random.random() < epsilon:
            # return index of max Q value of state
            return np.argmax(self.Q_values[current_state])
        else:  # return random index
            return np.random.randint(4)

    # function that gets next location based on chosen move
    def get_next_location(self, this_state, a_index):
        new_state = this_state
        if self.actions[a_index] == 'UP' and this_state[0] > 0:
            direction = (new_state[0] - 1, new_state[1])
            new_state = direction

        elif self.actions[a_index] == 'RIGHT' and this_state[1] < self.width - 1:
            direction = (new_state[0], new_state[1] + 1)
            new_state = direction

        elif self.actions[a_index] == 'DOWN' and this_state[0] < self.height - 1:
            direction = (new_state[0] + 1, new_state[1])
            new_state = direction

        elif self.actions[a_index] == 'LEFT' and this_state[1] > 0:
            direction = (new_state[0], new_state[1] - 1)
            new_state = direction

        return new_state

    def get_optimum_policy(self, start_state):
        if start_state == self.end_state:
            return []

        else:
            current_state = start_state
            # append to optimum policy list
            optimum_policy = [current_state]
            # append to optimum policy dictionary
            self.policy[current_state] = self.actions[self.get_next_move(current_state, 1)]

            while not self.is_terminal_state(current_state):
                # obtain next move
                a_index = self.get_next_move(current_state, 1)
                # move to the next location on the path, and add the new location to policy list
                current_state = self.get_next_location(current_state, a_index)
                optimum_policy.append(current_state)
                self.policy[current_state] = self.actions[self.get_next_move(current_state, 1)]

            self.opt_policy = optimum_policy
            return optimum_policy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create Q Learning object
    the_Object = QAgent(width=3, height=3, start_state=(0, 0), end_state=(0, 2), mines_number=1)
    the_Object.algorithm()
