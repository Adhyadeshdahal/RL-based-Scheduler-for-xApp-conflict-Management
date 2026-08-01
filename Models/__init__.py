from config import ExperimentConfig
from Models.CDL import CDL
from Models.MLP import MLPInference


def nodeNames(env):
    params = [f"param{i}" for i in range(env.num_params)]
    kpis = [kpi.name for kpi in env.kpis]
    return params + kpis


def get_model(cfg: ExperimentConfig, env):
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    model_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "device": cfg.device,
        "cmi_threshold": cfg.model.cmi_threshold,
        "eval_tau": cfg.model.eval_tau,
        "eval_steps": cfg.train.eval_steps,
        "grad_clip": cfg.model.grad_clip,
        "generative_fc_dims": cfg.model.generative_fc_dims,
        "feature_fc_dims": cfg.model.feature_fc_dims,
        "lr": cfg.model.lr,
        "kpi_start": env.num_params,
        "node_names": nodeNames(env),
    }
    if cfg.model_kind == "cdl":
        return CDL(**model_kwargs)
    if cfg.model_kind == "mlp":
        return MLPInference(**model_kwargs)
    raise ValueError(f"Unsupported model kind: {cfg.model_kind}")
