#Train Test Environment and method
import torch
from datetime import datetime

SEED = 50000
IS_TRAIN  =   False# Set to False for evaluation only
IS_TEST   =  not IS_TRAIN # Set to True for testing with a smaller number of steps
USE_CMI = True #  to True to use CMI-based model, False to use MLP-based model
USE_MLP = not USE_CMI
TEST_BATCH_SIZE = 10 # Used to test the model with mse loss, this denotes the batch size for testing, set to 1 for testing with mse loss for 1 sample
ENVIRONMENT = "EnvironmentI" #or "EnvironmentI" | "EnvironmentII"
NUM_STEPS = 10

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{ENVIRONMENT}-{timestamp}"

TOTAL_STEPS              = 50000 #Total steps for training the model
INIT_STEPS               = 4000    # No of steps that uses random exploration only
MODEL_BASED_START        = 50000   # No of steps after which it uses model based policy for exploration switch from random to model-based.Setting it to 20k effectively ensures random exploration only.
INFERENCE_GRADIENT_STEPS = 1 
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
N_CANDIDATE = 256  # Increased for better sampling
N_TOP       = 128  # Adjusted proportionally
N_ITER      = 20   # Increased for better convergence

#FOR MPPI
N_SAMPLES = 2000  # Increased for higher quality
TEMPERATURE = 0.6 # Lower for greedier selection
NOISE_SIGMA = 0.1 # Added small noise for exploration

#FOR MCTS
N_SIMULATIONS   = 3000  # Increased for deeper search
USB_C = 1.5   # Slightly higher for more exploration


RESULT_DIR = f"rslts/"
if USE_MLP:
    RESULT_DIR += "MLP/"
elif USE_CMI:
    RESULT_DIR += "CMI/"

RESULT_DIR += f"{RUN_NAME}/"


MODEL_LOAD_NAME = MODEL_SAVE_NAME = f"{"CMI" if USE_CMI else "MLP"}-{ENVIRONMENT}_model.pt" 
CDL_LOAD_NAME = f"CMI-{ENVIRONMENT}_model.pt" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")