import argparse
import random

import numpy as np
import torch

from Environment.Environment_I import ORANEnvironment
from Environment.Environment_II import ORANEnvironment2
from Models.CDL import CDL
from Models.MLP import MLPInference
from Parameters import (
    CMI_THRESHOLD,
    DEVICE,
    EVAL_TAU,
    FEATURE_FC_DIMS,
    GENERATIVE_FC_DIMS,
    GRAD_CLIP,
)
from Policies.RandomPolicy import RandomPolicy


SEED = 500


def state_to_tensor(state_dict):
    params, kpis = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            params.append(state_dict[key])
        elif "kpi" in key:
            kpis.append(state_dict[key])
    return torch.from_numpy(np.concatenate(params + kpis)).float()


def state_with_candidate_param(env, state_batch, action_batch):
    candidate_state = state_batch.clone()
    action_np = action_batch.detach().cpu().numpy()

    for row, action in enumerate(action_np):
        param_id = int(action[0])
        raw_value = env.action_to_param(
            [param_id, int(action[1]), int(action[2])]
        )[1]
        low, high = env.params[param_id].get_threshold()
        norm_value = (raw_value - low) / (high - low) if high != low else 1.0
        candidate_state[row, param_id] = float(norm_value)

    return candidate_state


def node_names(env):
    return [f"param{i}" for i in range(env.num_params)] + [
        kpi.name for kpi in env.kpis
    ]


def make_env(name):
    if name == "EnvironmentI":
        return ORANEnvironment()
    if name == "EnvironmentII":
        return ORANEnvironment2()
    raise ValueError(f"Unknown environment: {name}")


def make_model(kind, env):
    kwargs = dict(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        device=DEVICE,
        cmi_threshold=CMI_THRESHOLD,
        eval_tau=EVAL_TAU,
        grad_clip=GRAD_CLIP,
        generative_fc_dims=GENERATIVE_FC_DIMS,
        feature_fc_dims=FEATURE_FC_DIMS,
        lr=1e-3,
        kpi_start=env.num_params,
        node_names=node_names(env),
    )
    if kind == "CMI":
        return CDL(**kwargs)
    if kind == "MLP":
        return MLPInference(**kwargs)
    raise ValueError(f"Unknown model kind: {kind}")


def collect_transitions(env, steps):
    random_policy = RandomPolicy(env.action_dim, env.action_space)
    states, actions, next_states, post_change_states = [], [], [], []

    for _ in range(steps):
        obs = env.reset()
        state = state_to_tensor(obs)
        action = random_policy.act()
        next_obs, _, done, _ = env.step(action)
        if done:
            next_obs = env.reset()

        zero_action = np.zeros(env.action_dim)
        post_change_obs, _, _, _ = env.step(zero_action)

        states.append(state)
        actions.append(torch.tensor(action, dtype=torch.float32))
        next_states.append(state_to_tensor(next_obs))
        post_change_states.append(state_to_tensor(post_change_obs))

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(next_states),
        torch.stack(post_change_states),
    )


def predict_mean(model, states, actions):
    dist = model.predictNextState(states.float().to(model.device),
                                  actions.float().to(model.device))
    return dist.mean.detach().cpu()


def mse_by_variant(model, env, states, actions, target_kpis):
    changed_states = state_with_candidate_param(env, states, actions)
    zero_actions = torch.zeros_like(actions)

    variants = {
        "original_state + actual_action": (states, actions),
        "changed_state + zero_action": (changed_states, zero_actions),
        "changed_state + actual_action": (changed_states, actions),
        "original_state + zero_action": (states, zero_actions),
    }

    results = {}
    for name, (variant_states, variant_actions) in variants.items():
        pred = predict_mean(model, variant_states, variant_actions)
        err = (pred - target_kpis) ** 2
        results[name] = {
            "mse": float(err.mean()),
            "per_kpi_mse": [float(v) for v in err.mean(dim=0)],
        }
    return results


def print_results(environment, model_kind, target, steps, results):
    print(f"\nEnvironment: {environment}")
    print(f"Model      : {model_kind}")
    print(f"Target     : {target}")
    print(f"Transitions: {steps}")
    print("\nVariant                              MSE")
    print("--------------------------------  --------")
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["mse"]):
        print(f"{name:<32}  {metrics['mse']:.6f}")

    best = min(results.items(), key=lambda item: item[1]["mse"])
    print(f"\nBest: {best[0]} ({best[1]['mse']:.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="EnvironmentII",
                        choices=["EnvironmentI", "EnvironmentII"])
    parser.add_argument("--model", default="CMI", choices=["CMI", "MLP"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--target", default="one-step",
                        choices=["one-step", "post-change"])
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env(args.environment)
    model = make_model(args.model, env)
    model_path = args.model_path or f"{args.model}-{args.environment}_model.pt"
    model.load_model(model_path)

    states, actions, next_states, post_change_states = collect_transitions(
        env, args.steps
    )
    if args.target == "one-step":
        target_kpis = next_states[:, env.num_params:]
    else:
        target_kpis = post_change_states[:, env.num_params:]

    results = mse_by_variant(model, env, states, actions, target_kpis)
    print_results(args.environment, args.model, args.target, args.steps, results)


if __name__ == "__main__":
    main()
