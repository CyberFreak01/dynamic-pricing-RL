import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy import stats

class TaxiFareEnvironment:
    def __init__(self, features_df):
        self.features = features_df
        self.clean_and_prepare_data()
        self.scaler = RobustScaler()
        self.scaler.fit(self.features[['demand', 'competition', 'avg_price', 'relative_demand']])
        
        # Store min and max values for later use
        self.min_values = self.features[['demand', 'competition', 'avg_price', 'relative_demand']].min()
        self.max_values = self.features[['demand', 'competition', 'avg_price', 'relative_demand']].max()
    
    def clean_and_prepare_data(self):
        # Replace infinities with NaN
        self.features = self.features.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN values
        self.features = self.features.dropna()
        
        # Remove outliers using Z-score
        for column in ['demand', 'competition', 'avg_price', 'relative_demand']:
            z_scores = stats.zscore(self.features[column])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3)
            self.features = self.features[filtered_entries]
        
        # Clip very large values to 99th percentile
        for column in ['demand', 'competition', 'avg_price', 'relative_demand']:
            upper_limit = self.features[column].quantile(0.99)
            self.features[column] = self.features[column].clip(upper=upper_limit)
        
        print(f"Data shape after cleaning: {self.features.shape}")
    
    def get_state(self, pu_location, do_location, hour, day_of_week):
        state = self.features[(self.features['PULocationID'] == pu_location) & 
                              (self.features['DOLocationID'] == do_location) & 
                              (self.features['hour'] == hour) & 
                              (self.features['day_of_week'] == day_of_week)]
        if len(state) == 0:
            return None
        return self.scaler.transform(state[['demand', 'competition', 'avg_price', 'relative_demand']])[0]
    
    def get_reward(self, state, action):
        # Denormalize the average price
        avg_price = self.scaler.inverse_transform(state.reshape(1, -1))[0][2]
        
        # Calculate the new price based on the action
        price = avg_price * (1 + action)
        
        # Denormalize the demand
        demand = self.scaler.inverse_transform(state.reshape(1, -1))[0][0]
        
        # Simple demand elasticity model
        demand_multiplier = max(0, 1 - abs(action))
        revenue = price * demand * demand_multiplier
        return revenue

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
    
    def get_action(self, state):
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state_key])
    
    def update_q_table(self, state, action, reward, next_state):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        self.exploration_rate *= self.exploration_decay
    
    def _get_state_key(self, state):
        return tuple(map(lambda x: int(x * 10), state))

def train_model(env, agent, episodes=10000):
    for episode in range(episodes):
        state = env.get_state(np.random.randint(1, 264), np.random.randint(1, 264), 
                              np.random.randint(0, 24), np.random.randint(0, 7))
        if state is None:
            continue
        
        done = False
        while not done:
            action = agent.get_action(state)
            reward = env.get_reward(state, (action - 5) / 10)  # Map action to price adjustment (-0.5 to 0.5)
            next_state = env.get_state(np.random.randint(1, 264), np.random.randint(1, 264), 
                                       np.random.randint(0, 24), np.random.randint(0, 7))
            if next_state is None:
                done = True
                continue
            
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            
            if episode % 1000 == 0:   # HUGGING FACE !!!!
                done = True
        
        if episode % 1000 == 0:
            print(f"Episode {episode} completed. Exploration rate: {agent.exploration_rate:.2f}")


def get_optimal_fare(pu_location, do_location, hour, day_of_week, env, agent):
    state = env.get_state(pu_location, do_location, hour, day_of_week)
    if state is None:
        return None
    action = agent.get_action(state)
    price_adjustment = (action - 5) / 10  # Map action back to price adjustment (-0.5 to 0.5)
    
    # Denormalize the average price
    avg_price = env.scaler.inverse_transform(state.reshape(1, -1))[0][2]
    optimal_fare = avg_price * (1 + price_adjustment)
    return optimal_fare
