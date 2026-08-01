from Environment.Environment_I import ORANEnvironment
from Environment.Environment_II import ORANEnvironment2
from Parameters import ENVIRONMENT


def get_env():
    env = None
    if ENVIRONMENT == "EnvironmentII":
        env = ORANEnvironment2()
    elif ENVIRONMENT == "EnvironmentI":
        env = ORANEnvironment()
    else:
        raise NameError(f"env{env} is not valid")
    return env
