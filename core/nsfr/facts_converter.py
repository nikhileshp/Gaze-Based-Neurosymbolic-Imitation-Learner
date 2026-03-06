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
        self.e = 0
        self.d = 0
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = device
        self.atom_groups = None  # Cache for atom grouping

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
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def convert(self, Z, G, B, gaze=None):
        batch_size = Z.size(0)

        V = torch.zeros((batch_size, len(G))).to(
            torch.float32).to(self.device)

        # ── Gaze processing ──────────────────────────────────────────────────
        if gaze is not None:
            gaze = gaze.to(self.device)

            # Case 1: Heatmap gaze (B, H, W) -> Compute integral image
            if gaze.dim() == 3:
                gaze_padded   = torch.nn.functional.pad(gaze, (1, 0, 1, 0))
                gaze_integral = gaze_padded.cumsum(dim=1).cumsum(dim=2)

                # Compute Gaze Sums for all objects via SAT lookup: (B, N_OBJ)
                B_size, N_OBJ, _ = Z.shape
                flat_Z = Z.reshape(B_size * N_OBJ, -1)

                flat_gaze = gaze_integral.unsqueeze(1).expand(
                    -1, N_OBJ, -1, -1
                ).reshape(B_size * N_OBJ, gaze_integral.shape[1], gaze_integral.shape[2])

                sx, sy = 84.0 / 160.0, 84.0 / 210.0
                cx, cy, w, h = flat_Z[:, 1], flat_Z[:, 2], flat_Z[:, 3], flat_Z[:, 4]
                x,  y  = ((cx - w / 2) * sx).long(), ((cy - h / 2) * sy).long()
                dw, dh = (w * sx).long().clamp(min=1), (h * sy).long().clamp(min=1)
                x1, y1 = x.clamp(0, 84), y.clamp(0, 84)
                x2, y2 = (x + dw).clamp(0, 84), (y + dh).clamp(0, 84)
                idx = torch.arange(B_size * N_OBJ, device=self.device)

                sums = (
                    flat_gaze[idx, y2, x2]
                    - flat_gaze[idx, y1, x2]
                    - flat_gaze[idx, y2, x1]
                    + flat_gaze[idx, y1, x1]
                ).clamp(min=0.0)

                # Zero out absent objects
                is_present = (Z[:, :, 0] > 0.5).float()          # (B, N_OBJ)
                sums       = sums * (flat_Z[:, 0] > 0.5).float()
                gaze_sums  = sums.reshape(B_size, N_OBJ)          # (B, N_OBJ)

               if getattr(self.vm, "unnormalized", False):
    # Keep raw SAT sums — no normalization
                    gaze_sums = torch.max(gaze_sums, is_present * 0.05)
                else:
                    # Normalized: max-normalize so peak attended object = 1.0
                    gaze_max  = gaze_sums.max(dim=1, keepdim=True).values.clamp(min=1e-8)
                    gaze_sums = gaze_sums / gaze_max
                    gaze_sums = torch.max(gaze_sums, is_present * 0.05)

                # Append per-object gaze score as last feature: Z becomes (B, N_OBJ, F+1)
                Z = torch.cat([Z, gaze_sums.unsqueeze(-1)], dim=-1)

            # Case 2: Precomputed gaze features (B, N_OBJ, 1) or other shapes
            else:
                if gaze.shape[-1] == 1 and gaze.dim() == 3:
                    Z = torch.cat([Z, gaze], dim=-1)
                else:
                    # Fallback: append 1.0s if shape is unexpected
                    gaze_tensor = torch.ones((batch_size, Z.size(1), 1), device=self.device)
                    Z = torch.cat([Z, gaze_tensor], dim=-1)

        # ── Group atoms by predicate (cached) ────────────────────────────────
        # G is static per model so we build this once. If G ever changes
        # (different environment), reset self.atom_groups = None to rebuild.
        if self.atom_groups is None:
            self.atom_groups  = {}
            self.atom_indices = {}
            for i, atom in enumerate(G):
                if type(atom.pred) == NeuralPredicate and i > 1:
                    pred_name = atom.pred.name
                    if pred_name not in self.atom_groups:
                        self.atom_groups[pred_name]  = []
                        self.atom_indices[pred_name] = []
                    self.atom_groups[pred_name].append(atom)
                    self.atom_indices[pred_name].append(i)

        # ── Batch evaluation ─────────────────────────────────────────────────
        # gaze=None: val_fns never receive the heatmap directly.
        # Gaze scaling is applied in valuation.py via Z[:, :, -1] scalar
        # (the column appended above). The guard arg.size(1) >= 8 in
        # valuation._call_val_fn ensures no scaling fires in the no-gaze case.
        for pred_name, atoms in self.atom_groups.items():
            vals    = self.vm.batch_forward(Z, pred_name, atoms, gaze=None, all_objects=None)
            indices = self.atom_indices[pred_name]
            V[:, indices] = vals

        # ── Background knowledge ──────────────────────────────────────────────
        for i, atom in enumerate(G):
            if atom in B:
                V[:, i] += 1.0
        V[:, 1] = torch.ones((batch_size,)).to(torch.float32).to(self.device)
        return V

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred