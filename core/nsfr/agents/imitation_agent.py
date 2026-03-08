import torch
import torch.nn as nn
from nsfr.common import get_nsfr_model
from core.utils.utils import get_primitive_action_map

# class ImitationAgent(nn.Module):
#     def __init__(self, env_name, rules, device, gaze_threshold=None):
#         super().__init__()
#         self.device = device
#         self.env_name = env_name
#         self.primitive_action_map = get_primitive_action_map(env_name)
#         self.num_actions = max(self.primitive_action_map.values()) + 1
        
#         print(f"Initializing NSFR model for {env_name}...", flush=True)
#         self.model = get_nsfr_model(env_name, rules, device=device, train=True, gaze_threshold=gaze_threshold)
#         print("NSFR model initialized.", flush=True)
            
    
#     def _aggregate_to_action_scores(self, probs):
#         """
#         Shared helper: aggregates rule-level probs (B, num_rules) → action scores (B, num_actions).
#         Uses max-aggregation over rules that map to the same primitive action.
#         Called by update(), predict(), and act() to avoid code duplication.
#         """
#         action_rule_probs = {idx: [] for idx in range(self.num_actions)}
#         prednames = self.model.get_prednames()

#         for i, pred in enumerate(prednames):
#             prefix = pred.split('_')[0]
#             if prefix in self.primitive_action_map:
#                 idx = self.primitive_action_map[prefix]
#                 action_rule_probs[idx].append(probs[:, i])

#         action_scores_list = []
#         for idx in range(self.num_actions):
#             if action_rule_probs[idx]:
#                 stacked = torch.stack(action_rule_probs[idx], dim=1)
#                 m, _ = torch.max(stacked, dim=1)
#                 action_scores_list.append(m)
#             else:
#                 action_scores_list.append(torch.zeros(probs.size(0), device=self.device))

#         return torch.stack(action_scores_list, dim=1)  # (B, num_actions)


#     def update(self, states, actions, gazes=None, vT=None, loss_type='nll'):
#         """
#         Forward pass + loss computation for one batch.

#         Args:
#             states:  Logic state tensor          (B, num_atoms)
#             actions: Ground-truth action indices  (B,)
#             gazes:   Gaze centre tensor           (B, 2) or None
#             vT:      Pre-computed valuation tensor (B, num_atoms) or None
#             loss_type: 'nll' or 'bce'

#         Returns:
#             loss          – scalar tensor (call .backward() on this)
#             probs         – rule-level valuations  (B, num_rules),  detached
#             action_scores – action-level scores    (B, num_actions), detached
#         """
#         if vT is not None:
#             probs = self.model(vT.to(self.device))
#         else:
#             probs = self.model(states, gazes)

#         action_scores = self._aggregate_to_action_scores(probs)  # (B, num_actions)

#         eps = 1e-10
#         if loss_type == 'bce':
#             target_matrix = torch.zeros_like(action_scores)
#             target_matrix.scatter_(1, actions.unsqueeze(1), 1.0)
#             loss = nn.BCELoss()(action_scores, target_matrix)
#         else:  # 'nll'
#             log_action_scores = torch.log(action_scores + eps)
#             loss = nn.NLLLoss()(log_action_scores, actions)

#         # Return detached copies so callers can compute accuracy without
#         # accidentally holding onto the computation graph.
#         return loss, probs.detach(), action_scores.detach()
    
#     def predict(self, states, gazes=None, vT=None):
#         """
#         Inference-only forward pass — no loss, no grad.
#         Use this in validation loops and anywhere you only need action scores.

#         Returns:
#             probs         – rule-level valuations  (B, num_rules)
#             action_scores – action-level scores    (B, num_actions)
#         """
#         with torch.no_grad():
#             if vT is not None:
#                 probs = self.model(vT.to(self.device))
#             else:
#                 probs = self.model(states, gazes)

#             action_scores = self._aggregate_to_action_scores(probs)

#         return probs, action_scores

#     # def update(self, states, actions, gazes=None, vT=None, loss_type='nll'):
#     #     """
#     #     Update the model using a batch of states and actions.
#     #     Args:
#     #         states: Tensor of logic states (batch_size, num_atoms)
#     #         actions: Tensor of action indices (batch_size)
#     #         gazes: Tensor of gaze centers (batch_size, 2) or None
#     #         vT: Pre-computed intermediate valuation tensor (batch_size, num_atoms) or None
#     #     """
#     #     if vT is not None:
#     #         # The model's forward() automatically detects if the input is V_0 (B, num_atoms)
#     #         # and skips the slow FactsConverter (perception), continuing securely into self.im (reasoning).
#     #         probs = self.model(vT.to(self.device))
#     #     else:
#     #         # Forward pass (perception + reasoning)
#     #         probs = self.model(states, gazes)
        
#     #     # 1. Aggregate rules into Actions using Max (Argmax Aggregation)
#     #     # We find the best rule valuation for each action class.
#     #     action_rule_probs = {idx: [] for idx in range(self.num_actions)}
#     #     prednames = self.model.get_prednames()

#     #     for i, pred in enumerate(prednames):
#     #         prefix = pred.split('_')[0]
#     #         if prefix in self.primitive_action_map:
#     #             idx = self.primitive_action_map[prefix]
#     #             action_rule_probs[idx].append(probs[:, i])
        
#     #     action_scores_list = []
#     #     for idx in range(self.num_actions):
#     #         if action_rule_probs[idx]:
#     #             stacked = torch.stack(action_rule_probs[idx], dim=1)
#     #             m, _ = torch.max(stacked, dim=1)
#     #             action_scores_list.append(m)
#     #         else:
#     #             # Use the batch size from the first available action prob or 0 if none
#     #             batch_size = probs.size(0)
#     #             action_scores_list.append(torch.zeros(batch_size, device=self.device))
        
#     #     action_scores = torch.stack(action_scores_list, dim=1) # (B, num_actions)
        
#     #     batch_size = action_scores.size(0)
#     #     eps = 1e-10
#     #     if loss_type == 'bce':
#     #        probs = probs.clamp(0.0, 1.0)  # Binary Cross Entropy: target action → 1.0, all others → 0.0
#     #         # Respects logical independence of rules (no softmax compression)
#     #         target_matrix = torch.zeros(batch_size, self.num_actions, device=self.device)
#     #         target_matrix.scatter_(1, actions.unsqueeze(1), 1.0)
#     #         loss = nn.BCELoss()(action_scores, target_matrix)
#     #     else:  # 'nll'
#     #         # NLL: log of aggregated action scores, NLLLoss over classes
#     #         log_action_scores = torch.log(action_scores + eps)
#     #         loss = nn.NLLLoss()(log_action_scores, actions)

#     #     return loss, probs, action_scores

#     # def predict(self, states, gazes=None, vT=None):
#     # """Returns (probs, action_scores) without computing loss. For eval/val loops."""
#     # with torch.no_grad():
#     #     if vT is not None:
#     #         probs = self.model(vT.to(self.device))
#     #     else:
#     #         probs = self.model(states, gazes)
        
#     #     action_rule_probs = {idx: [] for idx in range(self.num_actions)}
#     #     for i, pred in enumerate(self.model.get_prednames()):
#     #         prefix = pred.split('_')[0]
#     #         if prefix in self.primitive_action_map:
#     #             idx = self.primitive_action_map[prefix]
#     #             action_rule_probs[idx].append(probs[:, i])
        
#     #     action_scores_list = []
#     #     for idx in range(self.num_actions):
#     #         if action_rule_probs[idx]:
#     #             stacked = torch.stack(action_rule_probs[idx], dim=1)
#     #             m, _ = torch.max(stacked, dim=1)
#     #             action_scores_list.append(m)
#     #         else:
#     #             action_scores_list.append(torch.zeros(probs.size(0), device=self.device))
        
#     #     return probs, torch.stack(action_scores_list, dim=1)

#     def act(self, state, gaze=None):
#         """
#         Select an action for a single state (online inference).

#         Args:
#             state: Logic state tensor (1, num_atoms) or (num_atoms,)
#             gaze:  Gaze centre tensor (1, 2) or None

#         Returns:
#             predicate – primitive action name string (e.g. 'up', 'fire')
#         """
#         if state.dim() == 1:
#             state = state.unsqueeze(0)

#         _, action_scores = self.predict(state, gaze)
#         best_action_idx = action_scores[0].argmax().item()

#         # Map action index back to primitive action name
#         inv_map = {v: k for k, v in self.primitive_action_map.items()}
#         predicate = inv_map[best_action_idx]

#         # Debug logging (every 50 calls)
#         if not hasattr(self, '_act_count'):
#             self._act_count = 0
#         self._act_count += 1
#         if self._act_count % 50 == 0:
#             prednames = self.model.get_prednames()
#             with torch.no_grad():
#                 rule_probs = self.model(state)
#             top_vals, top_idxs = torch.topk(rule_probs[0], min(3, len(prednames)))
#             # top_rules = [f"{prednames[i]}: {v:.4f}" for v, i in zip(top_vals, top_idxs)]
#             # print(f"  [DEBUG Agent] Top Rules: {', '.join(top_rules)} -> action: {predicate}")

#         return predicate

#     def save(self, path):
#         torch.save(self.model.state_dict(), path)

#     def load(self, path):
#         self.model.load_state_dict(torch.load(path, map_location=self.device))



class ImitationAgent(nn.Module):
    def __init__(self, env_name, rules, device, gaze_threshold=None, unnormalized=False, visible_preds_only=False, alpha=0.1, aggregation_method='max'):
        super().__init__()
        self.device = device
        self.env_name = env_name
        self.aggregation_method = aggregation_method
        self.primitive_action_map = get_primitive_action_map(env_name)
        self.num_actions = max(self.primitive_action_map.values()) + 1

        print(f"Initializing NSFR model for {env_name}...", flush=True)
        self.model = get_nsfr_model(env_name, rules, device=device, train=True, gaze_threshold=gaze_threshold, unnormalized=unnormalized, visible_preds_only=visible_preds_only, alpha=alpha)
        print("NSFR model initialized.", flush=True)

    @property
    def unwrapped_model(self):
        """
        Returns the underlying NSFR model regardless of DataParallel wrapping.
        Use this whenever you need to access model attributes or methods
        (e.g. get_prednames(), atoms, prednames) that DataParallel doesn't
        proxy automatically.
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _aggregate_to_action_scores(self, probs):
        """
        Shared helper: aggregates rule-level probs (B, num_rules) → action scores (B, num_actions).
        Uses max-aggregation over rules that map to the same primitive action.
        Called by update(), predict(), and act() to avoid code duplication.
        """
        action_rule_probs = {idx: [] for idx in range(self.num_actions)}
        # unwrapped_model: works whether or not DataParallel is wrapping self.model
        prednames = self.unwrapped_model.get_prednames()

        for i, pred in enumerate(prednames):
            prefix = pred.split('_')[0]
            if prefix in self.primitive_action_map:
                idx = self.primitive_action_map[prefix]
                action_rule_probs[idx].append(probs[:, i])

        def softor(tensors, dim=0):
            # Compute soft-or: 1 - prod(1 - p_i)
            res = 1.0
            for t in tensors:
                res = res * (1.0 - t)
            return 1.0 - res

        def max_agg(tensors, dim=0):
            # Compute max aggregation
            stacked = torch.stack(tensors, dim=1)
            res, _ = torch.max(stacked, dim=1)
            return res

        agg_fn = softor if self.aggregation_method == 'softor' else max_agg

        action_scores_list = []
        debug_info = []
        for idx in range(self.num_actions):
            if action_rule_probs[idx]:
                # action_score = softor(rules_list, dim=0) # (B,)
                action_score = agg_fn(action_rule_probs[idx], dim=0)
                action_scores_list.append(action_score)
                
                # For debugging first batch element
                inv_map = {v: k for k, v in self.primitive_action_map.items()}
                action_name = inv_map.get(idx, f"idx_{idx}")
                
                # Find rule names for this action
                action_rules = [pred for pred in prednames if pred.split('_')[0] == action_name]
                
                for i, prob_tensor in enumerate(action_rule_probs[idx]):
                    if i < len(action_rules):
                        rule_name = action_rules[i]
                        p_val = prob_tensor[0].item()
                        if p_val > 0.05:
                            debug_info.append(f"{rule_name}: {p_val:.4f}")
            else:
                action_scores_list.append(torch.zeros(probs.size(0), device=probs.device))

        res = torch.stack(action_scores_list, dim=1)
        
        if debug_info:
            if not hasattr(self, '_last_log_step'):
                self._last_log_step = 0
            self._last_log_step += 1
            if self._last_log_step % 10 == 0:
                # print(f"[DEBUG Agent] Rule Probs: {', '.join(debug_info)}")
                best_idx = res[0].argmax().item()
                inv_map = {v: k for k, v in self.primitive_action_map.items()}
                # print(f"[DEBUG Agent] Best Action: {inv_map.get(best_idx)} (score: {res[0, best_idx]:.4f})")

        return res

    def update(self, states, actions, gazes=None, vT=None, loss_type='nll'):
        """
        Forward pass + loss computation for one batch.

        Args:
            states:    Logic state tensor           (B, num_atoms)
            actions:   Ground-truth action indices   (B,)
            gazes:     Gaze centre tensor            (B, 2) or None
            vT:        Pre-computed valuation tensor (B, num_atoms) or None
            loss_type: 'nll' or 'bce'

        Returns:
            loss          – scalar tensor (call .backward() on this)
            probs         – rule-level valuations   (B, num_rules),  detached
            action_scores – action-level scores     (B, num_actions), detached
        """
        if vT is not None:
            probs = self.model(vT.to(self.device))
        else:
            probs = self.model(states, gazes)
        probs = probs.clamp(0.0, 1.0)  # softor can produce out-of-range values on empty clauses
        action_scores = self._aggregate_to_action_scores(probs)  # (B, num_actions)

        eps = 1e-10
        # Add this block right before the loss_type check in update()
        # Remove once error is found
        # if torch.isnan(probs).any():
        #     print(f"NaN in probs! min={probs.min():.4f} max={probs.max():.4f}")
        #     raise ValueError("NaN in probs")
        # if torch.isnan(action_scores).any():
        #     print(f"NaN in action_scores! min={action_scores.min():.4f} max={action_scores.max():.4f}")
        #     raise ValueError("NaN in action_scores")
        # if actions.max() >= self.num_actions:
        #     print(f"Action out of bounds! max action={actions.max()} num_actions={self.num_actions}")
        #     raise ValueError("Action index out of bounds")
        # print(f"  [DEBUG] probs range: [{probs.min():.4f}, {probs.max():.4f}]")
        # print(f"  [DEBUG] action_scores range: [{action_scores.min():.4f}, {action_scores.max():.4f}]")
        # print(f"  [DEBUG] actions range: [{actions.min()}, {actions.max()}]")

        
        if loss_type == 'bce':
            target_matrix = torch.zeros_like(action_scores)
            target_matrix.scatter_(1, actions.unsqueeze(1), 1.0)
            loss = nn.BCELoss()(action_scores, target_matrix)
        else:  # 'nll'
            log_action_scores = torch.log(action_scores.clamp(min=eps))
            loss = nn.NLLLoss()(log_action_scores, actions)
        # Return detached copies so callers can compute accuracy without
        # accidentally holding onto the computation graph.
        return loss, probs.detach(), action_scores.detach()

    def predict(self, states, gazes=None, vT=None):
        """
        Inference-only forward pass — no loss, no grad.
        Use this in validation loops and anywhere you only need action scores.

        Returns:
            probs         – rule-level valuations  (B, num_rules)
            action_scores – action-level scores    (B, num_actions)
        """
        with torch.no_grad():
            if vT is not None:
                probs = self.unwrapped_model(vT.to(self.device))
            else:
                probs = self.unwrapped_model(states, gazes)

            action_scores = self._aggregate_to_action_scores(probs)

        return probs, action_scores

    def act(self, state, gaze=None):
        """
        Select an action for a single state (online inference).

        Args:
            state: Logic state tensor (1, num_atoms) or (num_atoms,)
            gaze:  Gaze centre tensor (1, 2) or None

        Returns:
            predicate – primitive action name string (e.g. 'up', 'fire')
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        _, action_scores = self.predict(state, gaze)
        best_action_idx = action_scores[0].argmax().item()

        # Map action index back to primitive action name
        inv_map = {v: k for k, v in self.primitive_action_map.items()}
        predicate = inv_map[best_action_idx]

        # Debug logging (every 50 calls)
        if not hasattr(self, '_act_count'):
            self._act_count = 0
        self._act_count += 1
        if self._act_count % 50 == 0:
            pass
            # prednames = self.unwrapped_model.get_prednames()
            # with torch.no_grad():
            #     rule_probs = self.unwrapped_model(state)
            # top_vals, top_idxs = torch.topk(rule_probs[0], min(3, len(prednames)))
            # top_rules = [f"{prednames[i]}: {v:.4f}" for v, i in zip(top_vals, top_idxs)]
            # print(f"  [DEBUG Agent] Top Rules: {', '.join(top_rules)} -> action: {predicate}")

        return predicate

    def save(self, path):
        # Always save the underlying model weights, not the DataParallel wrapper
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.unwrapped_model.state_dict(), path)

    def load(self, path):
        self.unwrapped_model.load_state_dict(torch.load(path, map_location=self.device))