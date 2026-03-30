#Train Test Environment and method
import torch
from datetime import datetime

IS_TRAIN  =  True # Set to False for evaluation only
IS_TEST   =  False # Set to True for testing with a smaller number of steps
USE_CMI = False # Set to True to use CMI-based model, False to use MLP-based model
USE_MLP = not USE_CMI
TEST_BATCH_SIZE = 1 # Used to test the model with mse loss, this denotes the batch size for testing, set to 1 for testing with mse loss for 1 sample
ENVIRONMENT = "EnvironmentII" #or "EnvironmentI" | "EnvironmentII"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{ENVIRONMENT}-{timestamp}"

TOTAL_STEPS              = 20001 #Total steps for training the model
INIT_STEPS               = 3000    # No of steps that uses random exploration only
MODEL_BASED_START        = 20000   # No of steps after which it uses model based policy for exploration switch from random to model-based.Setting it to 20k effectively ensures random exploration only.
INFERENCE_GRADIENT_STEPS = 1 #No of steps used by model based cem planner to converge to the value
BATCH_SIZE               = 128
PLOT_FREQ                = 500
EVAL_STEPS               = 10
CMI_THRESHOLD            = 0.2
EVAL_TAU                 = 0.99
GRAD_CLIP                = 10.0
GENERATIVE_FC_DIMS       = [64, 64]
FEATURE_FC_DIMS          = [64, 64]


#FOR CEM
N_HORIZON   = 1
N_CANDIDATE = 64
N_TOP       = 32
N_ITER      = 5

#FOR MPPI
N_SAMPLES = 500
TEMPERATURE = 1.0  # λ — lower = greedier towards best sample
NOISE_SIGMA = 2.0  # exploration width (in bin units)

#FOR MCTS
N_SIMULATIONS   = 300
USB_C = 1.41   # exploration constant (√2 is standard)


RESULT_DIR = f"rslts/"
if USE_MLP:
    RESULT_DIR += "MLP/"
elif USE_CMI:
    RESULT_DIR += "CMI/"

RESULT_DIR += f"{RUN_NAME}/"


MODEL_LOAD_NAME = MODEL_SAVE_NAME = f"{"CMI" if USE_CMI else "MLP"}-{ENVIRONMENT}_model.pt" 
CDL_LOAD_NAME = f"CMI-{ENVIRONMENT}_model.pt" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")