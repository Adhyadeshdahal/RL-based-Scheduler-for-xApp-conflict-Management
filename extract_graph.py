
from Environment import get_env
from Models import get_model
from Parameters import *

def main():
    if IS_TRAIN:
        print("\033[94mCannot show in Training Mode. Set IS_TRAIN to False to visualize the causal graph.\033[0m")
        return

    if USE_MLP:
        print("\033[94mMLP-based model does not have an explicit causal graph to visualize.\033[0m")
        return
    
    env = get_env()
    model = get_model(env)
    
    model.load_model(MODEL_LOAD_NAME)
    model.visualize_causal_graph()

if __name__ == "__main__":
    main()
    