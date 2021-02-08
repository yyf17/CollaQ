from .episode_buffer import ReplayBuffer


REGISTRY = {}

REGISTRY["ReplayBuffer"] = ReplayBuffer

# # for project QPLEX
# from .episode_buffer_QPLEX import ReplayBuffer as ReplayBuffer_QPLEX
# REGISTRY["ReplayBuffer_QPLEX"] = ReplayBuffer_QPLEX

# for project DOP

# from .episode_buffer_DOP import ReplayBuffer as ReplayBuffer_DOP
from .episode_buffer import Best_experience_Buffer
# REGISTRY["ReplayBuffer_DOP"] = ReplayBuffer_DOP
REGISTRY["Best_experience"] = Best_experience_Buffer
