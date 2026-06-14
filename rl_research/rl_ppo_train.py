from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)

from rl_pairs_train import OUT_DIR, PAIRS, TEST_PAIRS, TRAIN_PAIRS, VALIDATION_PAIRS, WARMUP, build_pair

COST_PER_UNIT_CHANGE = 0.0004
COST_ANNEAL_UPDATES = 300
ENTROPY_START = 0.03
ENTROPY_END = 0.002
ENTROPY_DECAY_UPDATES = 600
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP = 0.2
LEARNING_RATE = 3e-4
PPO_EPOCHS = 1
MINIBATCH = 32768
EPISODES_PER_UPDATE = 16
VAL_EVERY = 5
PROGRESS_PATH = OUT_DIR / "rl_ppo_progress.jsonl"
WEIGHTS_PATH = OUT_DIR / "rl_ppo_weights.pt"


class PolicyNet(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(64, 3)
        self.value_head = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)


def load_split(dates: list[str], holdout_days: int, mode: str = "pairs") -> tuple[list, list, list, list]:
    groups: dict[str, list] = {"train": [], "validation": [], "test": [], "test_pure": []}
    if mode == "time":
        validation_dates = {dates[-holdout_days - 1]}
        test_dates = set(dates[-holdout_days:])
        train_dates = set(dates) - validation_dates - test_dates
        for date in dates:
            for symbol in PAIRS:
                episode = build_pair(symbol, date)
                if episode is None:
                    continue
                entry = (f"{date}:{symbol}", episode)
                if date in train_dates:
                    groups["train"].append(entry)
                elif date in validation_dates:
                    groups["validation"].append(entry)
                else:
                    groups["test"].append(entry)
                    groups["test_pure"].append(entry)
        return tuple(groups[name] for name in ("train", "validation", "test", "test_pure"))
    early = dates[: len(dates) - holdout_days] if len(dates) > holdout_days else dates
    late = dates[len(early) :]
    for date in dates:
        for symbol in PAIRS:
            episode = build_pair(symbol, date)
            if episode is None:
                continue
            entry = (f"{date}:{symbol}", episode)
            if date in early and symbol in TRAIN_PAIRS:
                groups["train"].append(entry)
            if (date in early and symbol in VALIDATION_PAIRS) or (date in late and symbol in TRAIN_PAIRS):
                groups["validation"].append(entry)
            if date in late and symbol in (VALIDATION_PAIRS + TEST_PAIRS):
                groups["test"].append(entry)
            if date in late and symbol in TEST_PAIRS:
                groups["test_pure"].append(entry)
    return tuple(groups[name] for name in ("train", "validation", "test", "test_pure"))


def to_tensors(entries: list) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features = torch.tensor(
        np.stack([episode["features"][WARMUP:] for _, episode in entries]), dtype=torch.float32
    )
    perp = torch.tensor(np.stack([episode["perp"][WARMUP:] for _, episode in entries]), dtype=torch.float32)
    returns = torch.zeros_like(perp)
    returns[:, :-1] = perp[:, 1:] / perp[:, :-1] - 1
    names = [name for name, _ in entries]
    return features, returns, names


def episode_pnl(positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    changes = torch.zeros_like(positions)
    changes[:, 0] = positions[:, 0].abs()
    changes[:, 1:] = (positions[:, 1:] - positions[:, :-1]).abs()
    rewards = positions * returns - COST_PER_UNIT_CHANGE * changes
    return rewards.sum(dim=1) * 100


def deterministic_eval(model: PolicyNet, features: torch.Tensor, returns: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits, _ = model(features)
        positions = (logits.argmax(dim=-1) - 1).float()
        pnls = episode_pnl(positions, returns)
        activity = (positions != 0).float().mean()
    return pnls.mean().item(), activity.item()


def compute_gae(rewards: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    streams, horizon = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(streams)
    for t in range(horizon - 1, -1, -1):
        next_value = values[:, t + 1] if t + 1 < horizon else torch.zeros(streams)
        delta = rewards[:, t] + GAMMA * next_value - values[:, t]
        last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[:, t] = last_advantage
    return advantages, advantages + values


def emit(record: dict) -> None:
    record["time_ms"] = int(time.time() * 1000)
    with PROGRESS_PATH.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--holdout-days", type=int, default=3)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dates = sorted(date.strip() for date in args.dates.split(",") if date.strip())
    train, validation, test, test_pure = load_split(dates, args.holdout_days)
    print(f"episodes: train {len(train)}, val {len(validation)}, test {len(test)} (pure {len(test_pure)})", flush=True)
    train_features, train_returns, _ = to_tensors(train)
    val_features, val_returns, _ = to_tensors(validation)
    feature_count = train_features.shape[-1]
    model = PolicyNet(feature_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val = -1e18

    generator = torch.Generator().manual_seed(args.seed)
    for update in range(1, args.updates + 1):
        cost = COST_PER_UNIT_CHANGE * min(1.0, update / COST_ANNEAL_UPDATES)
        entropy_bonus = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * min(1.0, update / ENTROPY_DECAY_UPDATES)
        subset = torch.randperm(train_features.shape[0], generator=generator)[:EPISODES_PER_UPDATE]
        batch_features = train_features[subset]
        batch_returns = train_returns[subset]
        with torch.no_grad():
            logits, values = model(batch_features)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            positions = (actions - 1).float()
            changes = torch.zeros_like(positions)
            changes[:, 0] = positions[:, 0].abs()
            changes[:, 1:] = (positions[:, 1:] - positions[:, :-1]).abs()
            rewards = positions * batch_returns - cost * changes
            advantages, value_targets = compute_gae(rewards, values)
            sampled_pnl = (rewards.sum(dim=1) * 100).mean().item()

        flat_features = batch_features.reshape(-1, feature_count)
        flat_actions = actions.reshape(-1)
        flat_log_probs = log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        flat_targets = value_targets.reshape(-1)
        sample_count = flat_features.shape[0]
        entropy_value = 0.0
        for _ in range(PPO_EPOCHS):
            order = torch.randperm(sample_count)
            for start in range(0, sample_count, MINIBATCH):
                batch = order[start : start + MINIBATCH]
                logits_batch, values_batch = model(flat_features[batch])
                distribution_batch = torch.distributions.Categorical(logits=logits_batch)
                new_log_probs = distribution_batch.log_prob(flat_actions[batch])
                ratio = (new_log_probs - flat_log_probs[batch]).exp()
                advantage_batch = flat_advantages[batch]
                surrogate = torch.min(
                    ratio * advantage_batch,
                    ratio.clamp(1 - CLIP, 1 + CLIP) * advantage_batch,
                )
                entropy = distribution_batch.entropy().mean()
                value_loss = (values_batch - flat_targets[batch]).pow(2).mean()
                loss = -surrogate.mean() - entropy_bonus * entropy + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                entropy_value = entropy.item()

        record = {"update": update, "train_sampled_pnl": round(sampled_pnl, 4), "entropy": round(entropy_value, 4)}
        if update % VAL_EVERY == 0:
            val_pnl, val_activity = deterministic_eval(model, val_features, val_returns)
            record["val_pnl"] = round(val_pnl, 4)
            record["val_activity"] = round(val_activity, 4)
            if val_pnl > best_val:
                best_val = val_pnl
                torch.save(model.state_dict(), WEIGHTS_PATH)
                record["new_best"] = True
        emit(record)

    model.load_state_dict(torch.load(WEIGHTS_PATH))
    print("\nfinal evaluation of best-by-validation model:")
    for name, entries in (("train", train), ("validation", validation), ("test", test), ("test_pure", test_pure)):
        if not entries:
            continue
        features, returns, names = to_tensors(entries)
        pnl, activity = deterministic_eval(model, features, returns)
        print(f"{name:>10}: {pnl:+.3f}%/эпизод, в позиции {activity * 100:.0f}% времени")
    if test_pure:
        features, returns, names = to_tensors(test_pure)
        with torch.no_grad():
            logits, _ = model(features)
            positions = (logits.argmax(dim=-1) - 1).float()
            pnls = episode_pnl(positions, returns)
        for name, pnl in zip(names, pnls.tolist()):
            print(f"  {name}: {pnl:+.3f}%")


if __name__ == "__main__":
    main()
