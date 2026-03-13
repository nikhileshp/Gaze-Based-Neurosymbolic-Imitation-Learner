import os

from nsfr.facts_converter import FactsConverter
from nsfr.utils.logic import get_lang, build_infer_module
from nsfr.nsfr import NSFReasoner
from nsfr.valuation import ValuationModule


def get_nsfr_model(env_name: str, rules: str, device: str, train=False, gaze_threshold=None, unnormalized=False, visible_preds_only=False, alpha=0.1, target_diagonal=0.99, random_init=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    # Prioritize 'in/envs' for logic and valuation
    in_lang_base = f"in/envs/{env_name}/logic/"
    core_lang_base = f"core/envs/{env_name}/logic/"
    # Only use 'in/' logic if the specific ruleset (dataset) folder exists there
    lang_base_path = in_lang_base if os.path.exists(os.path.join(in_lang_base, rules)) else core_lang_base

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    in_val_fn = f"in/envs/{env_name}/valuation.py"
    core_val_fn = f"core/envs/{env_name}/valuation.py"
    # Valuation priority: only use 'in/' if the file actually exists
    val_fn_path = in_val_fn if os.path.exists(in_val_fn) else core_val_fn

    # Pass gaze_threshold to ValuationModule if it accepts it.
    # We need to update ValuationModule __init__ as well.
    val_module = ValuationModule(val_fn_path, lang, device, gaze_threshold=gaze_threshold, unnormalized=unnormalized, visible_preds_only=visible_preds_only, alpha=alpha)
    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device, env_name=env_name)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device, target_diagonal=target_diagonal, random_init=random_init)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return NSFR
