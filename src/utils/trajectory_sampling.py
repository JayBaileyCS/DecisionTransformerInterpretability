import numpy as np
from typing import Literal, Set, Tuple

OBJECT_TRANSLATIONS = {"key": 5., "ball": 6.}
TARGETS = {5.: {"correct": "key", "incorrect": "ball"}, 6.: {"correct": "ball", "incorrect": "key"}}

def get_filtered_trajectories(trajectories: dict[str, any], 
                              rtg: Tuple[Literal["gt", "lt", "eq"], float]=None, 
                              instruction: Literal["key", "ball"]=None, 
                              target: Literal["key", "ball"]=None) -> dict[str, any]:
    """Takes in a TrajectoryWriter object and returns a dictionary of filtered trajectories based on the rtg, instruction, and target.
    If an arg is None, it will not be filtered. Passing None to all args will just return the original trajectories.
    args:
        trajectories: {'observations': Tensor, 'actions': Tensor, 'rtgs': Tensor} object. 
        rtg (Optional): Tuple of comparison operator and rtg value.
        instruction (Optional): String representing the instruction object. Looks at the square immediately left of the agent at t=0.
        target (Optional): String representing the target object. Looks at the square immediately in front of the agent at t=-1."""
    valid_trajectories = {i for i in range(len(trajectories["observations"]))}
    if rtg is not None:
        valid_trajectories = valid_trajectories & filter_by_rtg(trajectories, rtg[1], rtg[0]) # Intersection of sets.
    if instruction is not None:
        valid_trajectories = valid_trajectories & filter_by_instruction(trajectories, instruction)
    if target is not None:
        valid_trajectories = valid_trajectories & filter_by_target(trajectories, target)
    for key in trajectories.keys():
        stacked_data = np.array([])
        if len(valid_trajectories) > 0:
            stacked_data = np.stack([trajectories[key][i] for i in valid_trajectories])
        trajectories[key] = stacked_data
    return trajectories


def filter_by_rtg(trajectories: dict[str, any], rtg: float=0.8, comparison: Literal["gt", "lt", "eq"]="gt") -> Set[int]:
    # Filter by rtg value.
    if comparison == "gt":
        valid_trajectories = [i for i in range(len(trajectories["rtgs"])) if trajectories["rtgs"][i][0].item() > rtg]
    elif comparison == "lt":
        valid_trajectories = [i for i in range(len(trajectories["rtgs"])) if trajectories["rtgs"][i][0].item() < rtg]
    else:
        valid_trajectories = [i for i in range(len(trajectories["rtgs"])) if trajectories["rtgs"][i][0].item() == rtg]
    return set(valid_trajectories)


def filter_by_instruction(trajectories: dict[str, any], instruction: Literal["key", "ball"]):
    # Square on direct left at first timestep.
    valid_trajectories = [i for i in range(len(trajectories["observations"])) if trajectories["observations"][i, 0, 2, 6, 0] == OBJECT_TRANSLATIONS[instruction]]
    return set(valid_trajectories)

def filter_by_target(trajectories: dict[str, any], target: Literal["key", "ball"], rtg_threshold: float=0.8):
    # The target should be wrong (not the instruction) if rtg is low, correct (same as instruction) if rtg is high.
    expected_targets = ["correct" if trajectories["rtgs"][i][0].item() > rtg_threshold else "incorrect" for i in range(len(trajectories["rtgs"]))]
    valid_trajectories = [i for i in range(len(trajectories["observations"])) if is_valid_target([trajectories["observations"][i, -1, 3, 5, 0], expected_targets[i]])]
    return set(valid_trajectories)

def is_valid_target(target: int, expected: Literal["correct", "incorrect"]):
    # Returns True if the target is the expected target based on the RTG value.
    return target in TARGETS and OBJECT_TRANSLATIONS[target] == TARGETS[target][expected]