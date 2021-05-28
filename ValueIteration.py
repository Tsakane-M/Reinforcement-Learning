import numpy as np
import random


class ValueIteration:
    def __init__(self, width=3, height=2, all_states=None, rewards=None, mines_number=0, start_state=(0, 0),
                 end_state=(0, 3), actions=None, landmines=None):
        self.width = width
        self.height = height

        self.start_state = start_state
        self.end_state = end_state
        self.mines_number = mines_number
        
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
# .....................................................................................................end of Class

    def value_iteration(self):
        print("Hello Sucker!")
        iterations = 0
        gamma = 0.8
        Theta = 0.005
        old_Values = self.V

        while True:
            biggest_difference = 0
            # For each state.
            for state in self.all_states:
                print(f'state: {state}')
                # store old value
                old_value = self.V[state]
                new_value = 0
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
                        print(f' value for action {action} is {value}')

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
                        print(f' value for action {action} is {value}')

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
                        print(f' value for action {action} is {value}')

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
                        print(f' value for action {action} is {value}')

                # Find max value of V(s) for the cell.
                print(f' Values list is {Values}')
                maximum = np.max(Values)

                # Save the max value of V(s) for that cell.

                print(f' maximum for state {state} is {maximum}\n')
                self.V[state] = maximum
                if state == self.end_state:
                    self.policy[state] = 'STAY'
                else:
                    self.policy[state] = self.actions[state][np.argmax(Values)]

                biggest_difference = max(biggest_difference, np.abs(old_value - self.V[state]))

            print(f'New Value Function: {self.V}')
            print(f'Policy: {self.policy}')

            if biggest_difference < Theta:
                print(f'Finished with {iterations} iterations.')
                break
            else:
                iterations += 1
























                    

                