import numpy as np
from ValueIteration import ValueIteration
# This is the driver file.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create ValueIteration object
    the_Object = ValueIteration(width=2, height=2, start_state=(0, 0), end_state=(1, 1), mines_number=1)

    # print(f' Actions: {the_Object.actions}')
    print(f'Rewards: {the_Object.rewards}')
    print(f'All states: {the_Object.all_states}')
    print(f'Landmines: {the_Object.landmines}')
    print(f'Policy: {the_Object.policy}')
    print(f'Value Function: {the_Object.V}')

    # compute Value Iteration Algorithm
    the_Object.value_iteration()

    print(f'driver file Record: {the_Object.reshaped_record}')

