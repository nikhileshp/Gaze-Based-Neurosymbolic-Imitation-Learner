from typing import Sequence, Dict, Any, Union
from abc import ABC
import inspect
import re

import torch
from torch import nn
from nsfr.fol.language import Language
from nsfr.fol.logic import Atom, Const
from nsfr.utils.common import load_module


class ValuationFunction(nn.Module, ABC):
    """Base class for valuation functions used inside valuation modules."""

    def __init__(self, pred_name: str):
        super().__init__()
        self.pred_name = pred_name

    def forward(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def bool2probs(self, bool_tensor: torch.Tensor) -> torch.Tensor:
        """Converts a Boolean tensor into a probability tensor by assigning
        probability 0.99 for True
        probability 0.01 for False."""
        return torch.where(bool_tensor, 0.99, 0.01)


class ValuationModule(nn.Module, ABC):
    """Turns logic state representations into valuated atom probabilities according to
    the environment-specific valuation functions.

    Args:
        val_fn_path: The path to the file containing the user-specified valuation functions.
    """

    lang: Language
    device: Union[torch.device, str]
    val_fns: Dict[str, ValuationFunction]  # predicate names to corresponding valuation fn

    def __init__(self, val_fn_path: str, lang: Language, device: Union[torch.device, str],
                 pretrained: bool = True, gaze_threshold=None):
        super().__init__()

        # Parse all valuation functions
        val_fn_module = load_module(val_fn_path)
        all_functions = inspect.getmembers(val_fn_module, inspect.isfunction)
        self.val_fns = {fn[0]: fn[1] for fn in all_functions}
        
        self.lang = lang
        self.device = device
        self.pretrained = pretrained
        self.gaze_threshold = gaze_threshold
        
        # Cache for term grounding (term.name -> index or onehot)
        self.term_cache = {}

    def forward(self, zs: torch.Tensor, atom: Atom, gaze: torch.Tensor = None):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representation (the output of the YOLO model).
                atom (atom): The target atom to compute its probability.
                gaze (tensor): The gaze center (batch_size, 2). Optional.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # args: the vectorized input evaluated by the value function
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        return self._call_val_fn(atom.pred.name, args, gaze)

    def batch_forward(self, zs: torch.Tensor, pred_name: str, atoms: Sequence[Atom], gaze: torch.Tensor = None, all_objects: torch.Tensor = None):
        """Convert object-centric representation to valuation tensors for a batch of atoms of the same predicate.
        
        Args:
            zs: (Batch, N_OBJ, Features)
            pred_name: str
            atoms: List of Atom objects (length N)
            gaze: (Batch, H, W) integral image or None
            all_objects: (Batch, N_OBJ, Features) — full logic state for gaze normalization
            
        Returns:
            (Batch, N) tensor of valuations
        """
        batch_size = zs.size(0)
        num_atoms = len(atoms)
        if num_atoms == 0:
            return torch.zeros(batch_size, 0).to(self.device)
            
        # 1. Gather Arguments: List of [Arg0_Tensor, Arg1_Tensor, ...]
        # Each ArgX_Tensor should be (Batch * N, Features)
        
        # Assume all atoms have same arity and term structure (guaranteed by predicate grouping)
        arity = len(atoms[0].terms)
        flat_args = []
        
        for i in range(arity):
            term_tensors = [self.ground_to_tensor(atom.terms[i], zs) for atom in atoms]
            # Stack: (Batch, N, F)
            stacked = torch.stack(term_tensors, dim=1)
            # Reshape: (Batch*N, F)
            flat = stacked.view(batch_size * num_atoms, -1)
            flat_args.append(flat)
            
        # 2. Expand Gaze if needed
        flat_gaze = None
        if gaze is not None:
            gaze = gaze.to(self.device)
            if len(gaze.shape) > 2:
                # gaze: (Batch, H, W)
                # expand to (Batch, N, H, W) -> (Batch*N, H, W)
                gaze_expanded = gaze.unsqueeze(1).expand(-1, num_atoms, -1, -1)
                flat_gaze = gaze_expanded.reshape(batch_size * num_atoms, gaze.shape[1], gaze.shape[2])
            else:
                # Point gaze: (Batch, 2)
                gaze_expanded = gaze.unsqueeze(1).expand(-1, num_atoms, -1)
                flat_gaze = gaze_expanded.reshape(batch_size * num_atoms, -1)

        # 3. Expand all_objects if needed: (Batch, N_OBJ, F) -> (Batch*num_atoms, N_OBJ, F)
        flat_all_objects = None
        if all_objects is not None:
            all_objects = all_objects.to(self.device)
            if all_objects.dim() == 3:
                ao_expanded = all_objects.unsqueeze(1).expand(-1, num_atoms, -1, -1)  # (B, N, N_OBJ, F)
                flat_all_objects = ao_expanded.reshape(batch_size * num_atoms, all_objects.size(1), all_objects.size(2))

        # 4. Call Valuation Function
        val_flat = self._call_val_fn(pred_name, flat_args, flat_gaze, flat_all_objects)
        
        # 5. Reshape back
        val = val_flat.view(batch_size, num_atoms)
        return val

    def _call_val_fn(self, pred_name, args, gaze, all_objects=None):
        try:
            val_fn = self.val_fns[pred_name]
        except KeyError as e:
            raise NotImplementedError(f"Missing implementation for valuation function '{pred_name}'.")

        sig = inspect.signature(val_fn)
        accepts_gaze = 'gaze' in sig.parameters
        accepts_all_objects = 'all_objects' in sig.parameters

        # Try to pass gaze map if available and function accepts it
        if gaze is not None and len(gaze.shape) > 2:
            try:
                if accepts_gaze and accepts_all_objects and all_objects is not None:
                    val = val_fn(*args, gaze=gaze, all_objects=all_objects)
                elif accepts_gaze:
                    val = val_fn(*args, gaze=gaze)
                else:
                    val = val_fn(*args)
            except Exception as e:
                # Fallback
                print(f"Error calling {pred_name} with gaze: {e}")
                val = val_fn(*args)
        else:
            val = val_fn(*args)

        # Gaze-based valuation scaling (Old Logic for points)
        if self.gaze_threshold is not None and gaze is not None and len(gaze.shape) == 2 and gaze.shape[1] == 2 and pred_name.startswith("visible_"):
             if len(args) > 0:
                obj_tensor = args[0]
                if obj_tensor.shape[0] == gaze.shape[0]:
                    obj_pos = obj_tensor[:, 1:3] 
                    dist = torch.norm(obj_pos - gaze, dim=1)
                    dist = torch.clamp(dist, min=1e-6)
                    scale = torch.clamp(self.gaze_threshold / dist, max=1.0)
                    val = val * scale
        
        return val

    def ground_to_tensor(self, const: Const, zs: torch.Tensor):
        """Ground constant (term) into tensor representations.

            Args:
                const (const): The term to be grounded.
                zs (tensor): The object-centric state representation.
        """
        # Check cache first
        if const.name in self.term_cache:
            cached_val = self.term_cache[const.name]
             # If it's an integer, it's an object index
            if isinstance(cached_val, int):
                return zs[:, cached_val]
            else:
                 # It's a param tuple for one-hot (index, length)
                return self.to_onehot_batch(cached_val[0], cached_val[1], zs.size(0))

        # Check if the constant name is in the reserved style, e.g., "obj0", "obj1", etc.
        result = re.match(r"obj(\d+)", const.name)  # Changed to match obj0, obj1, obj2, ...
        if result is not None:
            # The constant is an object constant
            obj_id = result[1]
            obj_index = int(obj_id)  # No need to subtract 1 since we now support obj0
            self.term_cache[const.name] = obj_index
            return zs[:, obj_index]

        elif const.dtype.name == 'object':
            obj_index = self.lang.term_index(const)
            self.term_cache[const.name] = obj_index
            return zs[:, obj_index]

        elif const.dtype.name == 'image':
            return zs

        else:
             # Compute params for one-hot and cache them
            params = self.get_onehot_params(const)
            self.term_cache[const.name] = params
            return self.to_onehot_batch(params[0], params[1], zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        elif term.dtype.name == 'type':
            return self.to_onehot_batch(self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name)),
                                        batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def get_onehot_params(self, term):
        """Get parameters (index, length) for one-hot encoding."""
        if term.dtype.name == 'color':
            return self.colors.index(term.name), len(self.colors)
        elif term.dtype.name == 'shape':
            return self.shapes.index(term.name), len(self.shapes)
        elif term.dtype.name == 'material':
            return self.materials.index(term.name), len(self.materials)
        elif term.dtype.name == 'size':
            return self.sizes.index(term.name), len(self.sizes)
        elif term.dtype.name == 'side':
            return self.sides.index(term.name), len(self.sides)
        elif term.dtype.name == 'type':
            return self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name))
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size."""
        onehot = torch.zeros(batch_size, length).to(self.device)
        onehot[:, i] = 1.0
        return onehot
