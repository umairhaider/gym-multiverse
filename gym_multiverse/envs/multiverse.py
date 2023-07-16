import gym
from gym import spaces
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class MultiverseGym(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment uses multiple GPT-2 models to generate text based on an initial prompt.
    """
    _tokenizers = []
    _models = []

    @classmethod
    def load_models_and_tokenizers(cls, model_names):
        """Class method to load models and tokenizers based on the given model names"""
        for model_name in model_names:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                model.to('cuda')
            cls._tokenizers.append(tokenizer)
            cls._models.append(model)
        return cls._models, cls._tokenizers

    def default_reward_policy(self, actions):
        """
        Default reward policy which rewards the agent for generating novel words.
        """
        words_in_prompt = set(self.initial_prompt.split())
        words_in_action = set([self.tokenizers[0].decode([action.item()]) for action in actions])
        return torch.tensor([1.0 if word not in words_in_prompt else 0.0 for word in words_in_action])
    
    def __init__(self, model_names=['gpt2', 'gpt2'], initial_prompt="Once upon a time", max_length=100, batch_size=1, reward_policy=None):
        """
        Initialize the environment with given model names, initial prompt, max length and batch size
        """
        super(MultiverseGym, self).__init__()

        # Load pre-trained model (weights) and tokenizer
        self.models, self.tokenizers = self.load_models_and_tokenizers(model_names)
        
        self.batch_size = batch_size

        # Define the reward policy function
        self.reward_policy = reward_policy if reward_policy is not None else self.default_reward_policy

        # Define action and observation space
        # They must be gym.spaces objects
        # An action is an integer which is the id of the token in the vocabulary
        # An observation is a sequence of integers representing tokens
        vocab_size = self.tokenizers[0].vocab_size
        self.action_space = spaces.Discrete(vocab_size)
        self.observation_space = spaces.Box(low=0, high=vocab_size, shape=(max_length, ), dtype=np.int32)

        self.initial_prompt = initial_prompt
        self.state = self.reset()

    def step(self, actions):
        """
        Execute one time step within the environment
        """
        self.state = torch.cat([self.state, actions], dim=1)

        # Generate the next token logits using each model and average the results
        next_token_logits = torch.mean(torch.stack([model(self.state).logits[:, -1, :] for model in self.models]), dim=0)

        rewards = self.reward_policy(actions)
        done = (self.state.shape[1] >= self.observation_space.shape[0]).all()
        
        # Include next_token_logits in the info dictionary
        info = {"next_token_logits": next_token_logits}

        return self.state, rewards, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        prompt_ids = [tokenizer.encode(self.initial_prompt, return_tensors='pt') for tokenizer in self.tokenizers]
        self.state = torch.mean(torch.stack(prompt_ids), dim=0).repeat((self.batch_size, 1))

        return self.state

    def render(self, mode='human'):
        """
        Render the environment to the screen
        """
        import matplotlib.pyplot as plt

        for i in range(self.state.shape[0]):
            # Decode the current state into text and print it
            text = [tokenizer.decode(self.state[i]) for tokenizer in self.tokenizers]
            print(f'Environment {i}: {text}')

            # Generate next token probabilities with each model
            outputs = [model(self.state[i].unsqueeze(0)) for model in self.models]
            predictions = [output.logits[:, -1, :] for output in outputs]

            # Convert the predictions to probabilities using softmax
            probabilities = [torch.nn.functional.softmax(prediction, dim=1).detach().numpy() for prediction in predictions]

            # Average the probabilities
            avg_probabilities = np.mean(probabilities, axis=0)

            # Get the probabilities and indices of the top 10 next words
            top_probabilities, top_indices = torch.topk(torch.tensor(avg_probabilities), 10)

            # Convert the token indices to words
            top_words = [self.tokenizers[0].decode([idx]) for idx in top_indices[0]]

            # Create a bar plot of the word probabilities
            plt.figure(i)
            plt.bar(top_words, top_probabilities[0])
            plt.show()
