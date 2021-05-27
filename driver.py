import numpy as np
from ValueIteration import ValueIteration
# This is the driver file.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create ValueIteration object
    the_Object = ValueIteration(width=4, height=5, start_state=(0, 0), end_state=(3, 3), mines_number=3)

    # compute Value Iteration Algorithm
    # the_Object.compute_algorithm(data)

    print(f' Rewards: {the_Object.rewards}')
    print(f' Landmines: {the_Object.landmines}')
