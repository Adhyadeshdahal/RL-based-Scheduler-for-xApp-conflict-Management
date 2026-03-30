from Algorithms import get_algorithms
from Models.CDL import CDL
from Parameters import *
from Environment import get_env
from Models.MLP import MLPInference
from Models import get_model
import numpy as np

def state_to_tensor(state_dict):
    kpi, param = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            param.append(state_dict[key])
        elif "kpi" in key:
            kpi.append(state_dict[key])
    return torch.from_numpy(np.concatenate(param + kpi)).float()

def paramToXApp(Xapps, param2id):
    param_to_xapp = {}

    for xapp in Xapps:
        for p in xapp.params:
            pid = param2id[p]

            if pid not in param_to_xapp:
                param_to_xapp[pid] = []

            param_to_xapp[pid].append(xapp)

    return param_to_xapp


def main():
    env = get_env()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    model = get_model(env)

    cdl_model = CDL(
        state_dim=state_dim,
        action_dim=action_dim,
        device=DEVICE,
        cmi_threshold=CMI_THRESHOLD,
        eval_tau=EVAL_TAU,
        grad_clip=GRAD_CLIP,
        generative_fc_dims=GENERATIVE_FC_DIMS,
        feature_fc_dims=FEATURE_FC_DIMS,
        lr=1e-3,
        kpi_start=env.num_params
    )

    try:
        cdl_model.load_model(CDL_LOAD_NAME)
    except FileNotFoundError:
        print("CDL model not found")
        return -1

    try:
        model.load_model(MODEL_LOAD_NAME)
    except FileNotFoundError:
        print("Inference model not found")
        return -1


    causal_graph = cdl_model.get_binary_graph()[:, :-1].cpu().detach().numpy()

    dependencies = {}

    for row in range(env.num_params, state_dim):
        kpi_dependency = []

        for col in range(env.num_params):
            if causal_graph[row, col] == 1:
                kpi_dependency.append((col, env.params[col]))

        dependencies[row] = kpi_dependency


    algorithms = get_algorithms(model, env)

    param2id = {param: idx for idx, param in enumerate(env.params)}
    param2xapp = paramToXApp(env.xapps, param2id)


    for kpi, dependency in dependencies.items():
        for param_id, param in dependency:

            xapps_in_conflict = param2xapp.get(param_id, [])

            if len(xapps_in_conflict) == 0:
                continue

            weights_per_xapps = [1] * len(xapps_in_conflict)
            scaling_term = 10

            env.reset()
            curr_state = env._get_state()
            curr_state = state_to_tensor(curr_state).to(DEVICE)

            results = []

            for algorithm in algorithms:
                result = algorithm.act(
                    current_state=curr_state,
                    conflict_param_index=param_id,
                    xapps_under_conflict=xapps_in_conflict,
                    weights_per_xapps=weights_per_xapps,
                    scaling_term=scaling_term
                )
                results.append((algorithm.name, result))


            print(f"KPI {kpi}, Param {param_id}, Action: {results}")


if __name__ == "__main__":
    main()