from Environment import get_env
from Models import get_model
from config import DEFAULT_CONFIG, ExperimentConfig


def main(cfg: ExperimentConfig = DEFAULT_CONFIG):
    if cfg.model_kind == "mlp":
        print("\033[94mMLP-based model does not have an explicit causal graph to visualize.\033[0m")
        return

    env = get_env(cfg)
    model = get_model(cfg, env)

    model.load_model(f"CMI-{cfg.environment}_model.pt")
    thres = model.visualize_cmi_heatmap()
    model.visualize_causal_graph(threshold=thres)


if __name__ == "__main__":
    main()
