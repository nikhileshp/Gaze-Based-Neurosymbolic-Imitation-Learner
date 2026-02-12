print("Starting beam_search.py imports...")
import argparse
import torch
import os
import json
from nsfr.utils.beam import get_nsfr_model
from nsfr.utils.logic import get_lang
from nsfr.mode_declaration import get_mode_declarations
from nsfr.clause_generator import ClauseGenerator

print("Checking device...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.terminated = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminated[:]
        del self.predictions[:]

    def load_buffer(self, args):
        current_path = os.path.dirname(__file__)
        # Try finding it in results/bs_data first as per user request context
        path = os.path.join(current_path, '../results/bs_data', args.d)
        if not os.path.exists(path):
            path = os.path.join(current_path, 'data', args.d)
        
        with open(path, 'r') as f:
            state_info = json.load(f)

        limit = 100
        self.actions = torch.tensor(state_info['actions'][:limit]).to(device)
        self.logic_states = torch.tensor(state_info['logic_states'][:limit]).to(device)
        self.neural_states = torch.tensor(state_info['neural_states'][:limit]).to(device)
        self.action_probs = torch.tensor(state_info['action_probs'][:limit]).to(device)
        self.logprobs = torch.tensor(state_info['logprobs'][:limit]).to(device)
        self.rewards = torch.tensor(state_info['reward'][:limit]).to(device)
        self.terminated = torch.tensor(state_info['terminated'][:limit]).to(device)
        self.predictions = torch.tensor(state_info['predictions'][:limit]).to(device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument('-r', "--rules", required=True, help="choose to root rules", dest='r',
                        choices=["getout_root", 'threefish_root', 'loot_root', 'seaquest_root'])
    parser.add_argument('-m', "--model", required=True, help="the game mode for beam-search", dest='m',
                        choices=['getout', 'threefish', 'loot', 'seaquest'])
    parser.add_argument('-t', "--t-beam", type=int, default=3, help="Number of rule expantion of clause generation.")
    parser.add_argument('-n', "--n-beam", type=int, default=8, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--s", type=int, default=1, help="The size of the logic program.")
    parser.add_argument('--scoring', type=bool, help='beam search rules with scored rule by trained ppo agent',
                        default=False, dest='scoring')
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d')
    parser.add_argument('--use-limited-consts', action='store_true',
                        help='Use consts_limited.txt instead of consts.txt to reduce memory usage')
    args = parser.parse_args()
    return args


def run():
    print("Entering run()...")
    args = get_args()
    print("Args parsed.")
    # load state info for searching if scoring
    if args.scoring:
        buffer = RolloutBuffer()
        print("Loading buffer...")
        buffer.load_buffer(args)
        print("Buffer loaded.")
    # writer = SummaryWriter(f"runs/{env_name}", purge_step=0)
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, '../nsfr/nsfr', 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, '../nsfr/nsfr', 'data/lang/')

    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.m, use_limited_consts=args.use_limited_consts)
    print(f"Loaded language. Clauses: {len(clauses)}, Atoms: {len(atoms)}")
    bk_clauses = []
    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device=device)  # torch.device('cpu'))
    print("NSFR model initialized.")
    mode_declarations = get_mode_declarations(args, lang)

    print('get mode_declarations')
    if args.scoring:
        cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, mode_declarations, buffer=buffer, device=device)
    else:
        cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, mode_declarations, device=device)

    print("Starting clause generation...")
    clauses = cgen.generate(clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
    print("====== ", len(clauses), " clauses are generated!! ======")
    
    # Save to file
    output_file = "clauses.txt"
    with open(output_file, "w") as f:
        for clause in clauses:
            f.write(str(clause) + "\n")
    print(f"Clauses saved to {output_file}")


if __name__ == "__main__":
    run()
