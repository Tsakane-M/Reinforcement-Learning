import numpy as np
import random


class ValueIteration:
    def __init__(self, width=5, height=4, all_states=None, rewards=None, mines_number=0, start_state=(0, 0),
                 end_state=(0, 0), actions=None, landmines=None):
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
                actions[i] = ('D0WN', 'UP', 'LEFT', 'RIGHT')
            self.actions = actions

        # Setup landmines
        self.landmines = []
        # delete start and end states from a our list of states
        temp = self.all_states
        for state in all_states:
            # remove state if equal to start or end states
            if state == start_state:
                temp.remove(state)
            if state == end_state:
                temp.remove(state)

        # Choose random states for landmines
        self.landmines = random.sample(temp, mines_number)

        # Define rewards dictionary for all states
        if rewards is None:
            rewards = {}
            # setup rewards for all states
            for i in all_states:
                if i == start_state:
                    rewards[i] = 0
                elif i == end_state:
                    rewards[i] = 100
                else:
                    rewards[i] = -1

            # setup landmines
            for i in self.landmines:
                rewards[i] = -100

        self.rewards = rewards


                    

                