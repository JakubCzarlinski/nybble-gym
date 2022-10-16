"""Anything related to controlling the robot cat."""
import math
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import jit

BOUND_ANGLE_DEG = 15
BOUND_ANGLE_RAD = np.deg2rad(BOUND_ANGLE_DEG)
STEP_ANGLE_DEG = 15 # Maximum angle delta per step
STEP_ANGLE_RAD = np.deg2rad(STEP_ANGLE_DEG)
JOINT_LENGTH = 0.05 # Meters
JOINT_LENGTH_SQUARED = JOINT_LENGTH**2

@jit(nopython=True)
def leg_ik(angle: float,
           length: float,
           offset: float,
           sign: Optional[int]=1,
           sign_bot: Optional[int]=1) -> Tuple[float, float]:
    """
    Returns each angle in the joint of the leg, to match the desired swing angle
    and leg extension (length). Uses the formula: a^2 = b^2 + c^2 - 2abcos(C)
    """
    # Prevent negatives angles and angles of small magnitudes
    length = min(max(length, JOINT_LENGTH * 1.0), JOINT_LENGTH * 1.6)
    length_squared = length**2

    # Inner angle alpha
    upper_cos = (length_squared)/(2*JOINT_LENGTH*length)
    upper_joint = np.arccos(upper_cos)*sign + angle

    # Inner angle beta
    lower_cos = (2 * JOINT_LENGTH_SQUARED - length_squared) / (2 * JOINT_LENGTH_SQUARED)
    lower_joint = sign*(sign_bot*(np.arccos(lower_cos) - np.pi)) + offset

    if math.isnan(upper_joint):
        print("Upper joint angle is nan")
        upper_joint = 0
    if math.isnan(lower_joint):
        print("Lower joint angle is nan")
        lower_joint = 0

    return upper_joint, lower_joint

@jit(nopython=True)
def joint_extension(length: float) -> float:
    """Returns the length of the leg joint, biased to high lengths."""
    return 2 - ((1 - length)/2)**2

@jit(nopython=True)
def compute_desired_leg_joint_angles(joint_angles: npt.NDArray[np.float32],
                                     action: npt.NDArray[np.float32],
                                     sim: Optional[bool]=True) -> npt.NDArray[np.float32]:
    """Compute the desired joint angles for each leg, given the action."""

    # Use IK to compute the new angles of each joint
    desired_left_front_angle = BOUND_ANGLE_RAD * action[0]
    desired_left_front_length = JOINT_LENGTH * joint_extension(action[1])

    desired_right_front_angle = BOUND_ANGLE_RAD * action[2]
    desired_right_front_length = JOINT_LENGTH * joint_extension(action[3])

    desired_left_rear_angle = BOUND_ANGLE_RAD * action[4]
    desired_left_rear_length = JOINT_LENGTH * joint_extension(action[5])

    desired_right_rear_angle = BOUND_ANGLE_RAD * action[6]
    desired_right_rear_length = JOINT_LENGTH * joint_extension(action[7])

    # Compute the new angles of the joints
    joint_angles[0:2] = leg_ik(
        angle=desired_right_rear_angle * (1 if sim else -1),
        length=desired_right_rear_length,
        offset=(np.pi/2 if sim else -np.pi/2), #-np.pi/2
        sign=(1 if sim else -1)
    )
    joint_angles[2:4] = leg_ik(
        angle=desired_right_front_angle * (1 if sim else -1), #+ (np.pi/8 if sim else -np.pi/8)),
        length=desired_right_front_length,
        offset=(np.pi/2 if sim else np.pi/2),
        sign=(-1 if sim else 1),
        sign_bot=(-1 if sim else 1),
    )
    joint_angles[4:6] = leg_ik(
        angle=desired_left_rear_angle * (-1 if sim else -1),
        length=desired_left_rear_length,
        offset=(-np.pi/2 if sim else -np.pi/2), #?
        sign=-1,
    )
    joint_angles[6:8] = leg_ik(
        angle=desired_left_front_angle * (-1 if sim else -1), #+ (-np.pi/8 if sim else -np.pi/8)),
        length=desired_left_front_length,
        offset=(np.pi/2 if sim else np.pi/2),
        sign=1,
    )

    return joint_angles

@jit(nopython=True)
def compute_desired_angles(action: npt.NDArray[np.float32],
                           sim: Optional[bool]=True) -> npt.NDArray[np.float32]:
    """"Adds the action vector to the revolute joints. Joint angles are
    clipped. `joint_angles` is changed to have the updated angles of the
    entire robot. The vector returned contains the only the revolute joints
    of the robot."""
    joint_angles = compute_desired_leg_joint_angles(np.zeros(8, dtype=np.float32), action, sim)
    return joint_angles

def get_quaternion_from_euler(rpy):
    """
    Convert an Euler angle to a quaternion.

    Input
        :param roll: The roll (rotation around x-axis) angle in degrees.
        :param pitch: The pitch (rotation around y-axis) angle in degrees.
        :param yaw: The yaw (rotation around z-axis) angle in degrees.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll = rpy[0] * np.pi / 180
    pitch = rpy[1] * np.pi / 180
    yaw = rpy[2] * np.pi / 180

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]
