import numpy as np
import random


class QAgent:
    def __init__(self, width=3, height=2, all_states=None, rewards=None, mines_number=0, start_state=(0, 0),
                 end_state=(0, 3), actions=None, landmines=None, alpha=0, gamma=0, Q_values=None ):
        self.width = width
        self.height = height
        self.gamma = 0.75

        self.start_state = start_state
        self.end_state = end_state
        self.mines_number = mines_number
        self.record = []
        self.reshaped_record = []
        self.iterations = 0

        # Define all states
        if all_states is None:
            all_states = []
            for i in range(height):
                for j in range(width):
                    all_states.append((i, j))
        self.all_states = all_states

        # Dictionary of possible actions.
        if actions is None:
            actions = []
            actions = ('DOWN', 'UP', 'LEFT', 'RIGHT')
            self.actions = actions
        print(f'Actions: {self.actions}')

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
        print(f'Policy: {self.policy}')

        # Define initial Q-Value Function
        Q_values = {}

        for state in self.all_states:
            # Set values in all to be 0
            Q_values[state] = 0
        self.Q_values = Q_values
        print(f'Q values: {self.Q_values}')

        for state in self.all_states:
            self.record.append(0)

    # .....................................................................................................end of Class

    def algorithm(self):
        # Copy the rewards matrix to new Matrix

        # Pick up a state randomly
        current_state = random.choice(self.all_states)
        print(f'Rewards: {self.rewards}')
        gamma = self.gamma

        print(self.get_random_starting_location())

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create Q Learning object
    the_Object = QAgent(width=3, height=3, start_state=(0, 0), end_state=(0, 2), mines_number=1)
    the_Object.algorithm()





