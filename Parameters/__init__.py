#Train Test Environment and method
import torch
from datetime import datetime
from Tests import get_envII_mean_std,get_envI_mean_std

IS_TRAIN  =   True# Set to False for evaluation only
IS_TEST   =  not IS_TRAIN # Set to True for testing with a smaller number of steps
SEED = 0 if IS_TEST else 45 # Set to 0 for testing, 45 for training
USE_CMI = True #  to True to use CMI-based model, False to use MLP-based model
USE_MLP = not USE_CMI
TEST_BATCH_SIZE = 10 # Used to test the model with mse loss, this denotes the batch size for testing, set to 1 for testing with mse loss for 1 sample
ENVIRONMENT = "EnvironmentI" #or "EnvironmentI" | "EnvironmentII"
NUM_STEPS = 30 # Number of steps to run in test mode, set to 10 for quick testing, increase for more thorough evaluation

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{ENVIRONMENT}-{timestamp}"

TOTAL_STEPS              = 20000 #Total steps for training the model
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

#                                   ENVIRONMENT II

ENVIRONMENT_II_MEAN_STDS = []
ENVIRONMENT_II_PARAM_RANGES = []
if IS_TEST:
    ENVIRONMENT_II_PARAM_RANGES =[
                                    (100, 150),
                                    (50, 100),
                                    (-30, 30),
                                    (-90, 90),
                                    (-30, -19),
                                    (-50, 150),
                                    (66, 87),
                                    (-200, 150),
                                ]
    ENVIRONMENT_II_MEAN_STDS = get_envII_mean_std(ENVIRONMENT_II_PARAM_RANGES, seed=SEED)
    
    if ENVIRONMENT == "EnvironmentII":
        print(f"ENVIRONMENT_II_PARAM_RANGES: {ENVIRONMENT_II_PARAM_RANGES}")
        print(f"ENVIRONMENT_II_MEAN_STDS: {ENVIRONMENT_II_MEAN_STDS}")

else:
    ENVIRONMENT_II_PARAM_RANGES =    [
                                        (-100, 100),
                                        (-10, 50),
                                        (-20, 20),
                                        (-60, 60),
                                        (-20, 20),
                                        (-50, 150),
                                        (-60, 65),
                                        (-100, 150),
                                    ]
    ENVIRONMENT_II_MEAN_STDS = get_envII_mean_std(ENVIRONMENT_II_PARAM_RANGES, seed=SEED)
    
    if ENVIRONMENT == "EnvironmentII":
        print(f"ENVIRONMENT_II_MEAN_STDS: {ENVIRONMENT_II_MEAN_STDS}")
        print(f"ENVIRONMENT_II_PARAM_RANGES: {ENVIRONMENT_II_PARAM_RANGES}")


#                                           ENVIRONMENT I

ENVIRONMENT_I_MEAN_STDS = []
ENVIRONMENT_I_PARAM_RANGES = []

if IS_TEST:
    ENVIRONMENT_I_PARAM_RANGES = [
            (-10, 310),  # P1
            (-10, 310),  # P2
            (-5, 8),    # P3
            (-7, 10),    # P4
            (-10, 13),    # P5
            (0, 3),    # P6
            (-3, 6),    # P7
        ]
    ENVIRONMENT_I_MEAN_STDS = get_envI_mean_std(ENVIRONMENT_I_PARAM_RANGES, seed=SEED)
    if ENVIRONMENT == "EnvironmentI":
        print(f"ENVIRONMENT_I_PARAM_RANGES: {ENVIRONMENT_I_PARAM_RANGES}")
        print(f"ENVIRONMENT_I_MEAN_STDS: {ENVIRONMENT_I_MEAN_STDS}")

else:
    ENVIRONMENT_I_PARAM_RANGES = [
        (0, 300),  # P1
        (0, 300),  # P2
        (0, 3),    # P3
        (0, 3),    # P4
        (0, 3),    # P5
        (0, 3),    # P6
        (0, 3),    # P7
    ]
    ENVIRONMENT_I_MEAN_STDS = get_envI_mean_std(ENVIRONMENT_I_PARAM_RANGES, seed=SEED)
    if ENVIRONMENT == "EnvironmentI":
        print(f"ENVIRONMENT_I_PARAM_RANGES: {ENVIRONMENT_I_PARAM_RANGES}")
        print(f"ENVIRONMENT_I_MEAN_STDS: {ENVIRONMENT_I_MEAN_STDS}")
