from Models.CDL import CDL
from Models.MLP import MLPInference
from Parameters import *

def get_model(env):
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    model = None
    if USE_CMI:
        model = CDL(
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
    elif USE_MLP:
        model = MLPInference(
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
    
    return model