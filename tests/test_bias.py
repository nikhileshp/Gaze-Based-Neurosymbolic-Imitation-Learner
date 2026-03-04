import torch
import warnings
warnings.filterwarnings('ignore')
from train_il import *
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
args = argparse.Namespace(
    env='seaquest', rules='new', dataset='data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt',
    data_path=None, epochs=1, batch_size=32, lr=0.01, seed=42, device='cuda', limit=500, num_workers=0,
    val_split=0.0, gaze_threshold=50.0, use_gaze=False, use_gazemap=False, gaze_model_path='',
    num_episodes=1, sort_by=None, valuation_path='models/nsfr/seaquest/_no_gaze/valuation.pt',
    eval_interval=1, eval_max_steps=200
)

dataset = PtDataset(args.dataset, num_episodes=args.num_episodes)
if args.limit:
    dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
env = NudgeBaseEnv.from_name(args.env, mode='logic')
agent = ImitationAgent(args.env, args.rules, device=args.device)
agent.to(args.device)

optimizer = torch.optim.RMSprop(agent.model.parameters(), lr=args.lr)
agent.model.train()
print('Evaluating before training...')
agent.model.eval()
state = env.reset()
counts_before = {}
total_reward_before = 0
for i in range(1000):
    logic_state, _ = state
    logic_state_tensor = torch.as_tensor(logic_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    predicate = agent.act(logic_state_tensor)
    counts_before[predicate] = counts_before.get(predicate, 0) + 1
    state, r, d = env.step(predicate)
    total_reward_before += r
    if d: state = env.reset()
print(f'Counts before: {counts_before} | Total Reward: {total_reward_before}')

for epoch in range(5):
    agent.model.train()
    total_loss = 0
    for batch in dataloader:
        s, a, g, e, st = batch
        s, a = s.to(args.device), a.to(args.device)
        loss, _ = agent.update(s, a, None)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}')

    agent.model.eval()
    state = env.reset()
    counts_after = {}
    total_reward_after = 0
    for i in range(1000):
        logic_state, _ = state
        logic_state_tensor = torch.as_tensor(logic_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        predicate = agent.act(logic_state_tensor)
        counts_after[predicate] = counts_after.get(predicate, 0) + 1
        state, r, d = env.step(predicate)
        total_reward_after += r
        if d: state = env.reset()
    print(f'Epoch {epoch+1} Counts: {counts_after} | Reward: {total_reward_after}')

