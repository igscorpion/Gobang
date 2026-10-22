"""Masked policy loss and signed n-step targets for zero-sum A2C."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


def masked_distribution(logits, legal_mask):
    """Normalize only legal actions, rejecting all-masked policy rows."""
    if logits.ndim != 2 or logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have matching [B, A] shapes")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(legal_mask.any(dim=-1).all()):
        raise ValueError("Every policy row must contain a legal action")
    if not bool(torch.isfinite(logits[legal_mask]).all()):
        raise ValueError("Legal action logits must be finite")
    return Categorical(logits=logits.masked_fill(~legal_mask, -torch.inf))


@torch.no_grad()
def signed_n_step_returns(rewards, dones, signs, bootstrap, valid, gamma=0.99):
    """Return detached [T, N] targets, respecting turns, endings and padding.

    ``signs`` is -1 when the next state changes player perspective (self-play)
    and +1 when a learner/opponent pair returns to the learner's perspective.
    Invalid trailing rows leave the running bootstrap untouched and output zero.
    A done row cuts the recurrence before a different episode can contribute.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    if rewards.ndim != 2 or not rewards.is_floating_point():
        raise ValueError("rewards must be a floating point [T, N] tensor")
    if any(tensor.shape != rewards.shape for tensor in (dones, signs, valid)):
        raise ValueError("rewards, dones, signs and valid must share [T, N] shape")
    if bootstrap.shape != rewards.shape[1:]:
        raise ValueError("bootstrap must have shape [N]")
    valid = valid.to(device=rewards.device, dtype=torch.bool)
    dones = dones.to(device=rewards.device, dtype=torch.bool)
    signs = signs.to(device=rewards.device, dtype=rewards.dtype)
    running = bootstrap.detach().to(device=rewards.device, dtype=rewards.dtype)
    if not bool(torch.isfinite(rewards[valid]).all()):
        raise ValueError("Valid rewards must be finite")
    if not bool(torch.isfinite(running).all()):
        raise ValueError("Bootstrap values must be finite")
    if not bool(((signs[valid] == 1) | (signs[valid] == -1)).all()):
        raise ValueError("Valid perspective signs must be +1 or -1")
    targets = torch.zeros_like(rewards)
    for step in range(rewards.shape[0] - 1, -1, -1):
        continuation = torch.where(dones[step], torch.zeros_like(running), running)
        target = rewards[step] + gamma * signs[step] * continuation
        running = torch.where(valid[step], target, running)
        targets[step] = torch.where(valid[step], running, torch.zeros_like(running))
    return targets


def a2c_loss(
    logits,
    values,
    actions,
    returns,
    legal_mask,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """One fresh-rollout A2C loss; targets and policy advantages are detached."""
    distribution = masked_distribution(logits, legal_mask)
    batch_shape = logits.shape[:1]
    if any(tensor.shape != batch_shape for tensor in (values, actions, returns)):
        raise ValueError("values, actions and returns must have shape [B]")
    if logits.shape[0] == 0:
        raise ValueError("Cannot compute an A2C loss for an empty rollout")
    if value_coef < 0.0 or entropy_coef < 0.0:
        raise ValueError("Loss coefficients must be nonnegative")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Predicted values must be finite")
    targets = returns.detach().to(device=values.device, dtype=values.dtype)
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Return targets must be finite")
    actions = actions.to(device=logits.device, dtype=torch.long)
    if bool(((actions < 0) | (actions >= logits.shape[1])).any()):
        raise ValueError("Sampled action is outside the policy action space")
    mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(mask.gather(1, actions.unsqueeze(-1)).all()):
        raise ValueError("Sampled action must be legal")
    advantage = targets - values
    policy_loss = -(distribution.log_prob(actions) * advantage.detach()).mean()
    value_loss = advantage.square().mean()
    entropy = distribution.entropy().mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "total_loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "advantage_mean": float(advantage.detach().mean()),
        "return_mean": float(targets.mean()),
    }
    return loss, metrics
