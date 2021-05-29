# Import packages
import numpy as np
from ValueIteration import ValueIteration

import matplotlib.pyplot as plt

from Animate import generateAnimat
# This is the driver file.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create ValueIteration object
    the_Object = ValueIteration(width=4, height=4, start_state=(0, 0), end_state=(3, 2), mines_number=2)

    # print(f' Actions: {the_Object.actions}')
    print(f'Rewards: {the_Object.rewards}')
    print(f'All states: {the_Object.all_states}')
    print(f'Landmines: {the_Object.landmines}')

    print(f'Value Function: {the_Object.V}')

    # compute Value Iteration Algorithm
    the_Object.value_iteration()
    print(f'Universal Policy: {the_Object.policy}')

    print(f'Records Array: {the_Object.reshaped_record}')

    records = the_Object.reshaped_record

    start_state = the_Object.start_state
    end_state = the_Object.end_state
    mines = the_Object.landmines
    opt_pol = the_Object.create_optimum_policy(the_Object.policy)
    print(f'driver optimum policy: {the_Object.policy}')

    anim, fig, ax = generateAnimat(records, start_state, end_state, mines=mines, opt_pol=opt_pol,
                                   start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=False,
                                   vmin=-10, vmax=150)

    plt.show()
