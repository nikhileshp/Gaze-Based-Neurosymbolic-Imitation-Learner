import torch
import torch.nn as nn
from nsfr.common import get_nsfr_model

class ImitationAgent(nn.Module):
    def __init__(self, env_name, rules, device, lr=0.001):
        super().__init__()
        self.device = device
        self.model = get_nsfr_model(env_name, rules, device=device, train=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.NLLLoss()

    def update(self, states, actions):
        """
        Update the model using a batch of states and actions.
        Args:
            states: Tensor of logic states (batch_size, num_atoms)
            actions: Tensor of action indices (batch_size)
        """
        # Forward pass
        probs = self.model(states)
        
        # Compute loss (NLL on log probabilities)
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        loss = self.loss_fn(log_probs, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def act(self, state):
        """
        Select an action for a given state (inference).
        Args:
            state: Logic state tensor (1, num_atoms) or (num_atoms)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            probs = self.model(state)
            action = torch.argmax(probs, dim=1)
        return action.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
