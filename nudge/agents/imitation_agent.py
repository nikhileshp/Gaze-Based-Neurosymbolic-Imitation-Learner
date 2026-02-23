import torch
import torch.nn as nn
from nsfr.common import get_nsfr_model

class ImitationAgent(nn.Module):
    def __init__(self, env_name, rules, device, lr=0.001, gaze_threshold=None):
        super().__init__()
        self.device = device
        self.model = get_nsfr_model(env_name, rules, device=device, train=True, gaze_threshold=gaze_threshold)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.NLLLoss()

    def update(self, states, actions, gazes=None):
        """
        Update the model using a batch of states and actions.
        Args:
            states: Tensor of logic states (batch_size, num_atoms)
            actions: Tensor of action indices (batch_size)
            gazes: Tensor of gaze centers (batch_size, 2) or None
        """
        from scripts.data_utils import PRIMITIVE_ACTION_MAP
        
        # Forward pass
        probs = self.model(states, gazes)
        
        # Aggregate rule probabilities into primitive actions
        B = probs.size(0)
        action_probs = torch.zeros(B, 6, device=self.device)
        for i, pred in enumerate(self.model.get_prednames()):
            prefix = pred.split('_')[0]
            if prefix in PRIMITIVE_ACTION_MAP:
                action_probs[:, PRIMITIVE_ACTION_MAP[prefix]] += probs[:, i]
                
        # Compute loss (NLL on log probabilities)
        log_probs = torch.log(action_probs + 1e-10)
        loss = self.loss_fn(log_probs, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def act(self, state, gaze=None):
        """
        Select an action for a given state (inference).
        Returns the primitive action name string (e.g., 'up', 'fire').
        Args:
            state: Logic state tensor (1, num_atoms) or (num_atoms)
            gaze: Gaze center tensor (1, 2) or None
        """
        from scripts.data_utils import PRIMITIVE_ACTION_MAP
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            probs = self.model(state, gaze)
            
            # Aggregate probabilities for the 6 primitive actions
            action_probs = torch.zeros(1, 6, device=self.device)
            prednames = self.model.get_prednames()
            for i, pred in enumerate(prednames):
                prefix = pred.split('_')[0]
                if prefix in PRIMITIVE_ACTION_MAP:
                    action_probs[0, PRIMITIVE_ACTION_MAP[prefix]] += probs[0, i]
            
            # DEBUG: Print probabilities occasionally
            # We can use a global counter or just check if it's been a while
            if not hasattr(self, '_act_count'): self._act_count = 0
            self._act_count += 1
            if self._act_count % 50 == 0:
                action_names = ['noop', 'fire', 'up', 'right', 'left', 'down']
                prob_str = " | ".join([f"{name}: {p:.3f}" for name, p in zip(action_names, action_probs[0])])
                print(f"  [DEBUG Agent] Probs: {prob_str}")
                # Also print the top 3 rules
                top_rule_vals, top_rule_idxs = torch.topk(probs[0], 3)
                top_rules = [f"{prednames[idx]}: {val:.3f}" for val, idx in zip(top_rule_vals, top_rule_idxs)]
                print(f"  [DEBUG Agent] Top Rules: {', '.join(top_rules)}")
            
            action_idx = torch.argmax(action_probs, dim=1).item()
            
            # Convert index back to name for environment compatibility
            inv_map = {v: k for k, v in PRIMITIVE_ACTION_MAP.items()}
            predicate = inv_map[action_idx]
            
        return predicate

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))