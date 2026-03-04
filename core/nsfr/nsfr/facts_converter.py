import torch
import torch.nn as nn
from .fol.logic import NeuralPredicate
from tqdm import tqdm


class FactsConverter(nn.Module):
    """
    FactsConverter converts the output fromt the perception module to the valuation vector.
    """

    def __init__(self, lang, valuation_module, device=None):
        super(FactsConverter, self).__init__()
        # self.e = perception_module.e
        self.e = 0
        #self.d = perception_module.d
        self.d =0
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = device
        self.atom_groups = None # Cache for atom grouping

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def forward(self, Z, G, B, gaze=None):
        return self.convert(Z, G, B, gaze=gaze)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype(self):
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def convert(self, Z, G, B, gaze=None):
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
        V = torch.zeros((batch_size, len(G))).to(
            torch.float32).to(self.device)
        # Pre-compute gaze integral image if applicable
        gaze_arg = None
        if gaze is not None:
             gaze = gaze.to(self.device)
             if gaze.dim() == 3: # (B, H, W) -> Compute integral
                  gaze_padded = torch.nn.functional.pad(gaze, (1, 0, 1, 0))
                  gaze_arg = gaze_padded.cumsum(dim=1).cumsum(dim=2)
             else:
                  gaze_arg = gaze

                  gaze_arg = gaze

        # 1. Group atoms by predicate (if not cached)
        # We assume G (list of atoms) is static for a given FactsConverter usage context.
        # If G changes, we'd need to invalidate cache. 
        # For safety, let's cache based on length of G or just recompute if None.
        # But G is usually the full set of atoms.
        
        if self.atom_groups is None:
             self.atom_groups = {}
             # Also keep track of indices for scatter
             self.atom_indices = {}
             
             for i, atom in enumerate(G):
                 if type(atom.pred) == NeuralPredicate and i > 1:
                     pred_name = atom.pred.name
                     if pred_name not in self.atom_groups:
                         self.atom_groups[pred_name] = []
                         self.atom_indices[pred_name] = []
                     self.atom_groups[pred_name].append(atom)
                     self.atom_indices[pred_name].append(i)

        # 2. Batch Evaluation
        for pred_name, atoms in self.atom_groups.items():
            # Get valuations: (Batch, N)
            vals = self.vm.batch_forward(Z, pred_name, atoms, gaze=gaze_arg, all_objects=Z)
            
            # Scatter back to V
            # indices: (N,)
            indices = self.atom_indices[pred_name]
            # We can't do V[:, indices] = vals directly if indices is list?
            # Yes we can if indices is list or long tensor.
            V[:, indices] = vals

        # 3. Handle Background Knowledge (B)
        # This part seems static, could also be optimized but it's fast enough.
        for i, atom in enumerate(G):
             if atom in B:
                 V[:, i] += 1.0
        V[:, 1] = torch.ones((batch_size,)).to(
            torch.float32).to(self.device)
        return V

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred
