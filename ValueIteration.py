# Import packages

import numpy as np
import random

import matplotlib.pyplot as plt
from Animate import generateAnimat
import sys


class ValueIteration:
    def __init__(self, width=3, height=2, all_states=None, rewards=None, mines_number=0, start_state=(0, 0),
                 end_state=(0, 3), actions=None, landmines=None, gamma=0.8):
        self.width = width
        self.height = height

        self.start_state = start_state
        self.end_state = end_state
        self.mines_number = mines_number
        self.record = []
        self.reshaped_record = []
        self.iterations = 0
        self.gamma = gamma
        
        # Define all states
        if all_states is None:
            all_states = []
            for i in range(height):
                for j in range(width):
                    all_states.append((i, j))
        self.all_states = all_states

        # Dictionary of possible actions.
        if actions is None:
            actions = {}
            for i in all_states:
                actions[i] = ('DOWN', 'UP', 'LEFT', 'RIGHT')
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
                    rewards[i] = -1

            # setup rewards for landmines
            for i in self.landmines:
                rewards[i] = -100

        self.rewards = rewards

        # Define an initial policy
        self.policy = {}
        for s in actions.keys():
            # select random policy
            self.policy[s] = np.random.choice(actions[s])
            
        # Define initial Value Function
        V = {}
        
        for state in self.all_states:
            # Set values in all to be 0
            V[state] = 0
        self.V = V

        for state in self.all_states:
            self.record.append(0)

# .....................................................................................................end of Class

    def value_iteration(self):

        self.iterations = 1
        gamma = self.gamma
        Theta = 1

        while True:
            biggest_difference = 0
            # For each state.
            for state in self.all_states:
                # store old value
                old_value = self.V[state]
                Values = [-1000 for i in range(4)]

                # For each Iteration UP , DOWN, LEFT , RIGHT
                for action in self.actions[state]:
                    # set default next state value
                    next_state = state


                    if action == 'DOWN':
                        # check if valid state
                        if (state[0] - 1) < 0:
                            value = -10000
                        else:
                            # create next state
                            next_state = [state[0] - 1, state[1]]
                            # Calculate the value of the V(state) for up using next state
                            value = self.rewards[state] + (gamma * self.V[tuple(next_state)])

                        # Save value in a list
                        Values[0] = value

                    if action == 'UP':
                        # check if valid state
                        if (state[0] + 1) > (self.height - 1):
                            value = -10000
                        else:
                            # create next state
                            next_state = [state[0] + 1, state[1]]
                            # Calculate the value of the V(state) for up using next state
                            value = self.rewards[state] + (gamma * self.V[tuple(next_state)])

                        # Save value in a list
                        Values[1] = value

                    if action == 'LEFT':
                        # check if valid state
                        if (state[1] - 1) < 0:
                            value = -10000
                        else:
                            # create next state
                            next_state = [state[0], state[1] - 1]
                            # Calculate the value of the V(state) for up using next state
                            value = self.rewards[state] + (gamma * self.V[tuple(next_state)])

                        # Save value in a list
                        Values[2] = value

                    if action == 'RIGHT':
                        # check if valid state
                        if (state[1] + 1) > (self.width - 1):
                            value = -10000
                        else:
                            # create next state
                            next_state = [state[0], state[1] + 1]
                            # Calculate the value of the V(state) for up using next state
                            value = self.rewards[state] + (gamma * self.V[tuple(next_state)])

                        # Save value in a list
                        Values[3] = value

                # Find max value of V(s) for the cell.
                maximum = np.max(Values)

                # Save the max value of V(s) for that cell.

                self.V[state] = maximum
                self.record.append(maximum)

                if state == self.end_state:
                    self.policy[state] = 'STAY'
                else:
                    self.policy[state] = self.actions[state][np.argmax(Values)]

                biggest_difference = max(biggest_difference, np.abs(old_value - self.V[state]))

            if biggest_difference < Theta:
                print(f'Finished with {self.iterations} iterations.')
                break
            else:
                self.iterations += 1

        self.create_reshaped_record(self.record)
        self.create_optimum_policy(self.policy)

    def create_reshaped_record(self, record):
        a_record = np.array(record)

        new = a_record.reshape(self.iterations+1, self.width, self.height)
        list1 = new.tolist()

        self.reshaped_record = list1

    def create_optimum_policy(self, policy):
        opt_policy = []
        for state in policy:
            if state == self.start_state:
                opt_policy.append(state)

                while True:
                    if policy[state] == 'RIGHT':
                        direction = (state[0], state[1]+1)
                        opt_policy.append(direction)
                        state = direction

                    elif policy[state] == 'LEFT':
                        direction = (state[0], state[1] - 1)
                        opt_policy.append(direction)
                        state = direction

                    elif policy[state] == 'UP':
                        direction = (state[0]+1, state[1])
                        opt_policy.append(direction)
                        state = direction

                    elif policy[state] == 'DOWN':
                        direction = (state[0] - 1, state[1])
                        opt_policy.append(direction)
                        state = direction

                    elif policy[state] == 'STAY':
                        direction = (state[0], state[1])
                        # opt_policy.append(direction)
                        break

        return opt_policy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # used to set grid width
    width = int(sys.argv[1])
    # used to set grid height
    height = int(sys.argv[2])
    # used to set start state x value
    start_x = 1000000
    # used to set start state y value
    start_y = 1000000
    # used to set end state x value
    end_x = 1000000
    # used to set end state y value
    end_y = 1000000
    # used to set gamma (default = 0.8)
    gamma = 0.8
    # used to set number of landmines (default = 3)
    k = 3

    argument = 3

    while argument < len(sys.argv):

        if sys.argv[argument] == "-start":
            start_x = int(sys.argv[argument+1])
            start_y = int(sys.argv[argument + 2])
            argument += 2

        elif sys.argv[argument] == "-end":
            end_x = int(sys.argv[argument+1])
            end_y = int(sys.argv[argument + 2])
            argument += 2

        elif sys.argv[argument] == "-k":
            k = int(sys.argv[argument+1])
            argument += 1

        elif sys.argv[argument] == "-gamma":
            gamma = float(sys.argv[argument+1])
            argument += 1

        else:
            argument += 1

    if end_x == 1000000 or end_y == 1000000 or start_x == 1000000 or start_y == 1000000:
        for i in range(1000):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)

            end_x = np.random.randint(width)
            end_y = np.random.randint(height)

            if start_x == start_y and end_x == end_y:
                continue
            else:
                break

    start_state = (start_y, start_x)
    end_state = (end_y, end_x)

    # create ValueIteration object
    the_Object = ValueIteration(width=width, height=height, start_state=start_state,
                                end_state=end_state, mines_number=k, gamma=gamma)

    print(f'Environment size: {the_Object.height} x {the_Object.width}')
    print(f'Start State: {the_Object.start_state}')
    print(f'End State: {the_Object.end_state}')
    print(f'Landmines: {the_Object.landmines}')

    # compute Value Iteration Algorithm
    the_Object.value_iteration()
    records = the_Object.reshaped_record
    start_state = the_Object.start_state
    end_state = the_Object.end_state
    mines = the_Object.landmines
    opt_pol = the_Object.create_optimum_policy(the_Object.policy)
    print(f'Optimum policy: {opt_pol}')

    anim, fig, ax = generateAnimat(records, start_state, end_state, mines=mines, opt_pol=opt_pol,
                                   start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=False,
                                   vmin=-10, vmax=150)

    plt.show()




















