import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)  # Fully connected layer

    def forward(self, state):
        action_probs = torch.softmax(self.fc(state), dim=-1)  # Softmax to convert raw scores to probabilities
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Linear(state_dim, 1)  # Fully connected layer

    def forward(self, state):
        value = self.fc(state)  # Get value estimate for state
        return value

class PPOC_Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[1]
        self.action_dim = self.env.action_space.n

        # Initialize actor and critic
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        # Use different optimizers for actor and critic as they should update at different rates
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.002)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.clip_epsilon = 0.2  # Epsilon for clipping in the objective
        self.entropy_bonus = 0.01  # Coefficient for entropy bonus

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        action_dist = Categorical(action_probs)  # Use a categorical distribution for discrete action spaces
        action = action_dist.sample()  # Sample an action
        action_log_prob = action_dist.log_prob(action)  # Get log probability of the action
        return action.item(), action_log_prob.item()

    def update(self, states, actions, action_log_probs, rewards, next_states, dones):
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        with torch.no_grad():
            next_value = self.critic(next_states)  # Get critic's value estimate for next states

        returns = self.compute_returns(rewards, dones, next_value)  # Compute returns
        advantages = self.compute_advantages(rewards, dones, next_value, returns)  # Compute advantages

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate loss for actor
        action_probs = self.actor(states)
        curr_action_dist = Categorical(action_probs)
        curr_action_log_probs = curr_action_dist.log_prob(actions)

        ratio = torch.exp(curr_action_log_probs - action_log_probs)  # Importance sampling ratio
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        entropy = curr_action_dist.entropy().mean()  # Entropy for exploration
        actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_bonus * entropy  # Total actor loss

        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # Gradient clipping
        self.optimizer_actor.step()

        # Calculate loss for critic
        values = self.critic(states)
        critic_loss = nn.MSELoss()(values.squeeze(), returns)

        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # Gradient clipping
        self.optimizer_critic.step()

    def compute_returns(self, rewards, dones, next_value):
        # Compute the returns (discounted cumulative rewards) working backwards from the end of the episode
        returns = torch.zeros_like(rewards)
        running_return = next_value
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns

    def compute_advantages(self, rewards, dones, next_value, returns):
        # Compute the advantage (how much better an action was compared to the average action)
        advantages = torch.zeros_like(rewards)
        running_advantage = 0
        running_return = next_value
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * running_return * (1 - dones[t]) - running_return
            running_advantage = td_error + self.gamma * running_advantage * (1 - dones[t])
            running_return = returns[t]
            advantages[t] = running_advantage
        return advantages

    def train(self, num_episodes, save_every=10):
        # Training loop
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, action_log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.update([state], [action], [action_log_prob], [reward], [next_state], [done])  # Update actor and critic

                state = next_state

            print(f"Episode {episode+1}: Total reward = {total_reward}")

            # Save the trained model every 'save_every' episodes
            if (episode+1) % save_every == 0:
                torch.save(self.actor.state_dict(), f'trained_model_{episode+1}.pth')
                print(f"Model saved at episode {episode+1}.")

        # Save the trained model at the end of training
        torch.save(self.actor.state_dict(), 'models/trained_model.pth')
        print("Final model saved.")

    def load_model(self, model_path):
        # Load a trained model
        self.actor.load_state_dict(torch.load(model_path))
        print("Model loaded.")
