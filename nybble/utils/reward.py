from numba import jit
import numpy as np
import numpy.typing as npt

@jit('float32(float32[:], float32[:], float32[:], float32[::], float32[:])', nopython=True)
def reward_function(current_pos: npt.NDArray[np.float32],
                    previous_pos: npt.NDArray[np.float32],
                    current_state: npt.NDArray[np.float32],
                    previous_angles: npt.NDArray[np.float32],
                    desired_angles: npt.NDArray[np.float32]) -> np.float32:
    """Calculate reward based on the current state of the robot."""
    weights = np.array([50, -10, -5, 0, -0.005], dtype=np.float32)

    # Reward robot for moving forward
    forward_factor = current_pos[0] #- previous_pos[0] # possibly change this to - 0

    # Penalise robot for swaying to the side.
    horizontal_factor = abs(current_pos[1] - previous_pos[1])

    # Penalise if the robot is not standing upright.
    vertical_factor = 0
    if current_pos[2] < 0.075:
        vertical_factor = abs(current_pos[2] - previous_pos[2])

    # Penalise robot for not facing forward.
    orientation_factor = abs(np.fabs(current_state[0]))

    # Penalise robot for moving - this requires energy.
    angle_factor = abs(sum(
        (desired_angles - previous_angles[0]) - (previous_angles[0] - previous_angles[1])
    ))

    factors = np.array([
        forward_factor,
        horizontal_factor,
        vertical_factor,
        orientation_factor,
        angle_factor,
    ], dtype=np.float32)

    return np.dot(factors, weights)
