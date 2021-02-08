REGISTRY = {}
# for project CollaQ
from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner

REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner

# for project QPLEX
# from .episode_runner_QPLEX import EpisodeRunner_QPLEX
# from .parallel_runner_QPLEX import ParallelRunner_QPLEX

# REGISTRY["episode_QPLEX"] = EpisodeRunner_QPLEX
# REGISTRY["parallel_QPLEX"] = ParallelRunner

# for project RODE
# from .episode_runner_RODE import EpisodeRunner as EpisodeRunner_RODE
# from .parallel_runner_RODE import ParallelRunner as ParallelRunner_RODE

# REGISTRY["episode_RODE"] = EpisodeRunner_RODE
# REGISTRY["parallel_RODE"] = ParallelRunner_RODE   

# for project ROMA
# from .episode_runner_ROMA import EpisodeRunner as EpisodeRunner_ROMA
# from .parallel_runner_ROMA import ParallelRunner as ParallelRunner_ROMA

# REGISTRY["episode_ROMA"] = EpisodeRunner_ROMA
# REGISTRY["parallel_ROMA"] = ParallelRunner_ROMA

# for project NDQ
# from .episode_runner_NDQ import EpisodeRunner as EpisodeRunner_NDQ
# from .parallel_runner_NDQ import ParallelRunner as ParallelRunner_NDQ
from .episode_runner_full import EpisodeRunner_full
from .parallel_runner_x import ParallelRunner_x

# REGISTRY["episode_NDQ"] = EpisodeRunner_NDQ
# REGISTRY["parallel_NDQ"] = ParallelRunner_NDQ
REGISTRY["episode_full"] = EpisodeRunner_full
REGISTRY["parallel_x"] = ParallelRunner_x

# for project MAVEN
# from .parallel_runner_MAVEN import ParallelRunner as ParallelRunner_MAVEN
# REGISTRY["parallel_MAVEN"] = ParallelRunner_MAVEN