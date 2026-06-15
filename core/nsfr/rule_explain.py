"""Per-clause rule attribution for the play GUIs.

For each clause in an :class:`NSFReasoner`'s program, find the *winning* grounding
(the head-grounding + substitution whose body conjunction is largest) and report the
clause's probability together with that grounding's body atoms and their values.

This is the data behind the GUI's rule panel: "the ground predicates that make the
rule get that probability". It is pure (no pygame) and depends only on the reasoner's
``clauses``, ``atoms``, inference tensor ``im.I`` and the initial valuation ``V_0``.

Padding note: the inference tensor ``I`` has shape ``(C, G, S, L)`` mapping
clause x head-grounding x substitution x body-position -> atom index. Atom index 0 is
the FALSE atom and index 1 is the TRUE atom; short/invalid groundings are padded with
TRUE, which yields a body product of 1.0. Such pure-padding groundings must be masked
out before the argmax, or they would always win.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Reserved atom indices in the NSFR inference tensor.
FALSE_ATOM_IDX = 0
TRUE_ATOM_IDX = 1


@dataclass(frozen=True)
class RuleAttribution:
    """One clause's contribution this step.

    Attributes:
        clause_idx: Index into ``nsfr.clauses`` (program order).
        head: Clause head predicate name (e.g. ``"up_to_diver"``).
        prob: Product of the winning grounding's body-atom values, in [0, 1].
        body: ``[(str(atom), value), ...]`` for the winning grounding's real body
            atoms (padding TRUE/FALSE atoms excluded), in body order.
    """

    clause_idx: int
    head: str
    prob: float
    body: List[Tuple[str, float]]


def _valuation_row(nsfr) -> torch.Tensor:
    """Return the initial valuation V_0 for batch row 0, or ``None`` if unavailable."""
    v0 = getattr(nsfr, "V_0", None)
    if v0 is None or isinstance(v0, list) or not torch.is_tensor(v0):
        return None
    if v0.ndim != 2 or v0.size(0) == 0:
        return None
    return v0[0].detach()


def rule_attributions(nsfr) -> List[RuleAttribution]:
    """Compute per-clause attributions for the current valuation.

    Args:
        nsfr: An :class:`NSFReasoner` after a forward pass (so ``V_0`` is populated).

    Returns:
        One :class:`RuleAttribution` per clause, in program (clause) order. Returns an
        empty list if the reasoner is not in a state that can be explained (no ``im.I``
        or no valuation yet).
    """
    clauses = getattr(nsfr, "clauses", None)
    atoms = getattr(nsfr, "atoms", None)
    im = getattr(nsfr, "im", None)
    I = getattr(im, "I", None) if im is not None else None
    v0 = _valuation_row(nsfr)

    if not clauses or atoms is None or I is None or v0 is None:
        logger.debug("rule_attributions: reasoner not in an explainable state.")
        return []

    I = I.to(v0.device)
    out: List[RuleAttribution] = []

    with torch.no_grad():
        for c, clause in enumerate(clauses):
            head = clause.head.pred.name
            I_c = I[c]  # (G, S, L) atom indices
            vals = v0[I_c]  # (G, S, L) gathered valuations
            body_conj = vals.prod(dim=-1)  # (G, S)

            # A grounding is "real" only if it references at least one non-padding
            # body atom; pure-padding groundings give product 1.0 and must be ignored.
            real = (I_c > TRUE_ATOM_IDX).any(dim=-1)  # (G, S)
            prod_masked = body_conj.masked_fill(~real, -1.0)
            max_prod = float(prod_masked.max().item())
            if max_prod < 0:
                # No real grounding for this clause; fall back to head valuation.
                out.append(RuleAttribution(c, head, _head_fallback(nsfr, head), []))
                continue

            # Pick the max-product grounding; when several tie (e.g. all ~0 because the
            # rule doesn't fire), prefer the one whose body atoms have the highest total
            # value -> an actually-present object rather than an arbitrary empty slot.
            cand = prod_masked >= (max_prod - 1e-6)
            sum_vals = vals.sum(dim=-1).masked_fill(~cand, float("-inf"))
            flat_idx = int(sum_vals.argmax().item())

            g_star, s_star = divmod(flat_idx, I_c.size(1))
            prob = float(body_conj[g_star, s_star].item())
            body_indices = I_c[g_star, s_star]  # (L,)
            body = [
                (str(atoms[idx]), float(v0[idx].item()))
                for idx in body_indices.tolist()
                if idx > TRUE_ATOM_IDX
            ]
            out.append(RuleAttribution(c, head, prob, body))

    return out


def _head_fallback(nsfr, head: str) -> float:
    """Best-effort head valuation when a clause has no firing grounding."""
    try:
        return float(nsfr.get_predicate_valuation(head, initial_valuation=False))
    except Exception:  # noqa: BLE001 - display path must never crash the GUI
        return 0.0


def order_for_display(
    attributions: List[RuleAttribution], taken_head: str
) -> List[RuleAttribution]:
    """Reorder so the taken clause is first, the rest stay in program order.

    The "taken" clause is the highest-probability clause whose head matches
    ``taken_head`` (the executed action). If no head matches, order is unchanged.
    """
    if not attributions:
        return attributions
    candidates = [a for a in attributions if a.head == taken_head]
    if not candidates:
        return list(attributions)
    taken = max(candidates, key=lambda a: a.prob)
    rest = [a for a in attributions if a is not taken]
    return [taken] + rest
