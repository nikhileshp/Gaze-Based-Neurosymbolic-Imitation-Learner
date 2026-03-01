import torch
import torch.nn as nn
from nsfr.common import get_nsfr_model

class ImitationAgent(nn.Module):
    def __init__(self, env_name, rules, device, gaze_threshold=None):
        super().__init__()
        self.device = device
        print(f"Initializing NSFR model for {env_name}...", flush=True)
        self.model = get_nsfr_model(env_name, rules, device=device, train=True, gaze_threshold=gaze_threshold)
        print("NSFR model initialized.", flush=True)

    def update(self, states, actions, gazes=None, vT=None):
        """
        Update the model using a batch of states and actions.
        Args:
            states: Tensor of logic states (batch_size, num_atoms)
            actions: Tensor of action indices (batch_size)
            gazes: Tensor of gaze centers (batch_size, 2) or None
            vT: Pre-computed intermediate valuation tensor (batch_size, num_atoms) or None
        """
        from scripts.data_utils import PRIMITIVE_ACTION_MAP
        
        if vT is not None:
            # The model's forward() automatically detects if the input is V_0 (B, num_atoms)
            # and skips the slow FactsConverter (perception), continuing securely into self.im (reasoning).
            probs = self.model(vT.to(self.device))
        else:
            # Forward pass (perception + reasoning)
            probs = self.model(states, gazes)
        
        # 1. Aggregate rules into Actions using Max (Argmax Aggregation)
        # We find the best rule valuation for each action class.
        action_rule_probs = {idx: [] for idx in range(6)}
        prednames = self.model.get_prednames()

        for i, pred in enumerate(prednames):
            prefix = pred.split('_')[0]
            if prefix in PRIMITIVE_ACTION_MAP:
                idx = PRIMITIVE_ACTION_MAP[prefix]
                action_rule_probs[idx].append(probs[:, i])
        
        action_scores_list = []
        for idx in range(6):
            if action_rule_probs[idx]:
                stacked = torch.stack(action_rule_probs[idx], dim=1)
                m, _ = torch.max(stacked, dim=1)
                action_scores_list.append(m)
            else:
                # Use the batch size from the first available action prob or 0 if none
                batch_size = probs.size(0)
                action_scores_list.append(torch.zeros(batch_size, device=self.device))
        
        action_scores = torch.stack(action_scores_list, dim=1) # (B, 6)
        
        # Independent Binary Cross Entropy
        # We want target action to be 1.0, and ALL other actions to be penalized towards 0.0
        # This avoids Softmax which breaks logic independence, but still provides false-action penalty.
        batch_size = action_scores.size(0)
        target_matrix = torch.zeros(batch_size, 6, device=self.device)
        target_matrix.scatter_(1, actions.unsqueeze(1), 1.0)
        
        # Compute BCELoss over all actions independently
        bce_loss = nn.BCELoss(reduction='none')(action_scores, target_matrix)
        
        # We can sum the BCE components per sample so it evaluates properly per-experience
        sample_losses = bce_loss.sum(dim=1)
        loss = sample_losses.mean()
        
        return loss, sample_losses

    def act(self, state, gaze=None):
        """
        Select an action for a given state (inference).
        Returns the primitive action name string (e.g., 'up', 'fire').
        Args:
            state: Logic state tensor (1, num_atoms) or (num_atoms)
            gaze: Gaze center tensor (1, 2) or None
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Forward pass to get rule valuations V_T restricted to heads
            # probs: (1, num_rules)
            probs = self.model(state, gaze)
            
            # NOTE: Removed softmax! rule valuations are already in [0, 1].
            # Softmax on rule valuations squashes the signal and is logically incorrect.
            
            # Pick the rule with the highest valuation directly
            prednames = self.model.get_prednames()
            best_rule_idx = torch.argmax(probs[0]).item()
            best_rule = prednames[best_rule_idx]
            predicate = best_rule.split('_')[0]  # primitive action prefix

            # DEBUG: Print the winning rule occasionally
            if not hasattr(self, '_act_count'): self._act_count = 0
            self._act_count += 1
            if self._act_count % 50 == 0:
                top_rule_vals, top_rule_idxs = torch.topk(probs[0], min(3, len(prednames)))
                top_rules = [f"{prednames[idx]}: {val:.4f}" for val, idx in zip(top_rule_vals, top_rule_idxs)]
                # print(f"  [DEBUG Agent] Top Rules: {', '.join(top_rules)} -> action: {predicate}")
            
        return predicate

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))