from dataclasses import dataclass
from typing import Dict

from common.settings import Settings
from mapping.keyframe import KeyFrame


@dataclass
class OptimizationSettings:
    """ OptimizationSettings is a simple container for parameters for the optimizer
    """

    stage: int
    num_iterations: int = 1
    fix_poses: bool = False  # Fix the map, only optimize poses
    fix_sigma_mlp: bool = False
    fix_rgb_mlp: bool = False


class Optimizer:
    """ The Optimizer module is used to run iterations of the CLONeR Optimization.

    The KeyFrameManager supplies the Optimizer with a window of KeyFrame objects,
    which the Optimizer then uses to draw samples and iterate the optimization
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    # Run one or more iterations of the optimizer, as specified by the
    # @p optimization_settings using the @p keyframe_window as the set of
    # keyframes.
    def iterate_optimizer(self,
                          keyframe_window: Dict[KeyFrame, int],
                          optimization_settings: OptimizationSettings) -> float:
        pass

    # For a given @p keyframe, compute the loss for the stage given by @p stage.
    def keyframe_forward_pass(self, keyframe: KeyFrame, stage: int) -> float:
        pass
