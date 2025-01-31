import pandas as pd
import numpy as np
from utils.core import *
from tabulate import tabulate

# Usage
features = pd.read_csv('processed_features.csv')
env = TaxiFareEnvironment(features)
agent = QLearningAgent(state_size=10000, action_size=11)  # 11 actions for -50% to +50% price adjustments
train_model(env, agent) 

# Save the trained Q-table
np.save('q_table.npy', agent.q_table)
print("Q-table saved to 'q_table.npy'")

# Input Source Location ID, Destination Location ID, hours, days of week
pu_location = 4  #132       #4
do_location = 90   #130     #90
hour = 12           #12     #12
day_of_week = 2     #3      #2

optimal_fare = get_optimal_fare(pu_location, do_location, hour, day_of_week, env, agent)

if optimal_fare is not None:
    print(f"Optimal fare: ${optimal_fare:.2f}")
else:
    print("Unable to calculate optimal fare (No rides available in dataset at the time)")

def print_q_table(agent):
    q_table_df = pd.DataFrame.from_dict(agent.q_table, orient='index')
    
    # Change action labels to -5, -4, ..., 0, 1, 2
    action_labels = [-5 + i for i in range(agent.action_size)]
    q_table_df.columns = action_labels
    
    # Adding state labels
    q_table_df.index = [f'State {i}' for i in range(len(q_table_df.index))]
    
    print("\nQ-Table:")
    print(tabulate(q_table_df, headers='keys', tablefmt='psql', showindex=True))


print_q_table(agent)
