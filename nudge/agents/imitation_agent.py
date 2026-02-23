import torch
import torch.nn as nn
from nsfr.common import get_nsfr_model

class ImitationAgent(nn.Module):
    def __init__(self, env_name, rules, device, lr=0.001, gaze_threshold=None):
        super().__init__()
        self.device = device
        print(f"Initializing NSFR model for {env_name}...", flush=True)
        self.model = get_nsfr_model(env_name, rules, device=device, train=True, gaze_threshold=gaze_threshold)
        print("NSFR model initialized.", flush=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.NLLLoss()

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
            # Skip the forward pass and use pre-computed valuations
            probs = self.model.get_predictions(vT.to(self.device), prednames=self.model.prednames)
        else:
            # Forward pass (perception + reasoning)
            probs = self.model(states, gazes)
        
        # Apply softmax across all rules to make them compete
        B = probs.size(0)
        # 1. Aggregate rules into Actions using Mean (fixes Rule-Count Bias)
        # The user wants "divide it only by up actions", implying we normalize by rule count per action class.
        action_scores = torch.zeros(B, 6, device=self.device)
        action_rule_counts = torch.zeros(6, device=self.device)
        
        for i, pred in enumerate(self.model.get_prednames()):
            prefix = pred.split('_')[0]
            if prefix in PRIMITIVE_ACTION_MAP:
                idx = PRIMITIVE_ACTION_MAP[prefix]
                action_scores[:, idx] += probs[:, i]
                action_rule_counts[idx] += 1
                
        # Handle actions with no rules (clamp to 1 to avoid /0)
        action_rule_counts = action_rule_counts.clamp(min=1)
        action_averages = action_scores / action_rule_counts.unsqueeze(0)
        
        # 2. Convert to Action Probability Distribution
        # We normalize the averages so they sum to 1.0 across all 6 actions.
        # This allows P(UP) to reach 1.0 if its rules are firing, even if it has many rules.
        action_probs = action_averages / (action_averages.sum(dim=1, keepdim=True) + 1e-10)
        
        # 3. Compute NLL losses
        eps = 1e-10
        log_action_probs = torch.log(action_probs + eps)
        sample_losses = nn.NLLLoss(reduction='none')(log_action_probs, actions)
        
        # 4. Class-Balanced Batch Loss
        # Normalizes the gradient signal such that each action class in the batch has equal weight.
        batch_loss = 0.0
        unique_batch_actions = torch.unique(actions)
        for action_idx in unique_batch_actions:
            mask = (actions == action_idx)
            count = mask.sum().float()
            class_loss = sample_losses[mask].sum() / count
            batch_loss += class_loss
            
        loss = batch_loss / len(unique_batch_actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), sample_losses.detach().cpu().numpy()

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
            probs = self.model(state, gaze)
            
            # Apply softmax across all rules to make them compete
            probs = torch.softmax(probs, dim=1)
            
            # Pick the rule with the highest probability directly (argmax over rules)
            prednames = self.model.get_prednames()
            best_rule_idx = torch.argmax(probs[0]).item()
            best_rule = prednames[best_rule_idx]
            predicate = best_rule.split('_')[0]  # primitive action prefix

            # DEBUG: Print the winning rule occasionally
            if not hasattr(self, '_act_count'): self._act_count = 0
            self._act_count += 1
            if self._act_count % 50 == 0:
                top_rule_vals, top_rule_idxs = torch.topk(probs[0], 3)
                top_rules = [f"{prednames[idx]}: {val:.3f}" for val, idx in zip(top_rule_vals, top_rule_idxs)]
                print(f"  [DEBUG Agent] Top Rules: {', '.join(top_rules)} -> action: {predicate}")
            
        return predicate

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))