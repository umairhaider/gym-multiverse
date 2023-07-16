from agents.ppoc_agent import PPOC_Agent
from envs.multiverse import MultiverseGym

# Define the list of models to load
model_names = ['gpt2', 'gpt2-medium', 'gpt2-large']

# Create an instance of the MultiverseGym environment
env = MultiverseGym(model_names)

# Create an instance of the PPOC_Agent
agent = PPOC_Agent(env)

# Load a saved model
agent.load_model('models/trained_model.pth')

# Perform inference using the loaded model
state = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Total reward with loaded model: {total_reward}")
