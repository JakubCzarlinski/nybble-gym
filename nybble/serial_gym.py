"""Create a custom gym environment for the Nybble cat robot."""
from typing import Sequence, Tuple
import numpy as np
import numpy.typing as npt

from gym import Env
from gym import spaces
from sklearn.preprocessing import normalize

from .utils import controls
#from .utils.reward import reward_function
from .utils.serial_communication import Connection

## Hyper Params
MAX_EPISODE_LEN = 40  # Number of steps for one training episode
ACTION_HISTORY = 10
USE_GYRO = False

class SerialGym(Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        """ Initialize the environment."""
        # Number of time steps the environment has been running for.
        self.step_counter = 0
        
        self.conn = Connection(gyro=USE_GYRO) # USE_GYRO

        # The action space contains the 8 joint angles
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # The observation space are the torso roll, pitch and the joint angles and a history of the last X actions
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8 * ACTION_HISTORY + (6 if USE_GYRO else 0),), dtype=np.float32)

        self.robot_uid: int = 0
        self.joint_ids: Sequence[int] = []
        self.action_history = np.array([], dtype=np.float32)

        # Robot state consists of the orientation and velocity of the robot
        self.robot_state: npt.NDArray[np.float32] = np.zeros(6, dtype=np.float32)

    def step(self,
             action: npt.NDArray[np.float32],
             ) -> Tuple[npt.NDArray[np.float32], np.float32, bool, dict]:
        """ Perform one step in the simulation. `action` is a vector of values -1 <= x <= 1 .
        """

        # Keep track of the last 5 joint angle states. This is used as part of the observation.
        self.action_history = np.append(self.action_history, action)
        self.action_history = np.delete(self.action_history, np.s_[0:8])

        # action = np.array([
        #     (self.step_counter % 3) - 1, -1,     # Angle of upper then lower left front leg
        #     (self.step_counter % 3) - 1, -1,     # Angle of upper then lower right front leg
        #     (self.step_counter % 3) - 1, -1,     # Angle of upper then lower left back leg
        #     (self.step_counter % 3) - 1, -1,     # Angle of upper then lower right back leg
        # ]).astype(np.float32)
        
        desired_joint_angles = controls.compute_desired_angles(action, False)
        
        self.set_joint_angles(desired_joint_angles)

        # 20hz
        #time.sleep(0)

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        self.robot_state = self.get_robot_state()

        # reward = reward_function(
        #     np.asarray(robot_pos, dtype=np.float32),
        #     np.asarray(prev_pos, dtype=np.float32),
        #     self.robot_state,
        #     self.joint_angles_history,
        #     desired_joint_angles,
        # )
        
        reward: np.float32 = np.float32(0.0)

        # Stop criteria of current learning episode: Number of steps or robot fell
        done = False
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN or self.is_fallen():
            reward += np.float32(self.step_counter - MAX_EPISODE_LEN)
            done = True

        # No debug info
        info = {}
        if USE_GYRO:
            observation = np.concatenate((self.robot_state, self.action_history))
        else:
            observation = self.action_history

        return observation, reward, done, info

    def reset(self):
        """Reset the simulation to its original state."""
        self.step_counter = 0
        
        #self.conn.alarm()

        action = np.array([
            0.0, 0.5,     # Angle of upper then lower left front leg
            0.0, 0.5,     # Angle of upper then lower right front leg
            0.0, 0.5,     # Angle of upper then lower left back leg
            0.0, 0.5,     # Angle of upper then lower right back leg
        ]).astype(np.float32)

        # Set initial joint angles with some random noise
        reset_pos = controls.compute_desired_angles(action, False)
        
        self.set_joint_angles(reset_pos)

        # Initialize robot state history with reset position for X steps. This is as if the robot
        # was standing still.
        self.robot_state = self.get_robot_state()
        self.action_history = np.tile(action, ACTION_HISTORY)
        if USE_GYRO:
            observation = np.concatenate((self.robot_state, self.action_history))
        else:
            observation = self.action_history

        return observation

    def get_robot_state(self) -> npt.NDArray[np.float32]:
        """Create a state vector of the robot."""
        
        IMU_values = self.conn.get_IMU()
        
        robot_orientation = controls.get_quaternion_from_euler(IMU_values[0:3])
        
        robot_velocity = np.asarray(IMU_values[3:6]) / 20 # 20 is a guess, in hindsight this might be acceleration
        robot_velocity = robot_velocity[0:2]
        robot_velocity_norm = normalize(robot_velocity.reshape(-1,1))

        return np.concatenate((
            robot_orientation,
            robot_velocity_norm.reshape(1,-1)[0],  # type: ignore
        )).astype(np.float32)
        
    def get_joint_angles(self) -> npt.NDArray[np.float32]:
        angles = self.conn.get_joint_angles()
        return np.asarray(
             angles[0:4] + angles[8:],
             dtype=object,
         ).astype(np.float32) * np.pi / 180
        
    def set_joint_angles(self, angles: npt.NDArray[np.float32]):
        angle_map = ((np.asarray([0, 0, 0, 0, 0, 0, 0, 0,
            angles[6], angles[2], angles[0], angles[4], 
            angles[7], angles[3], angles[1], angles[5]])).astype(np.float32) * 180 / np.pi).tolist()
        angle_map = [int(x) for x in angle_map]
        print("Setting joint angles: " + str(angle_map))
        self.conn.set_joint_angles(angle_map)

    def render(self, mode='human'):
        pass

    def close(self):
        self.conn.close()

    def is_fallen(self) -> bool:
        """ Check if robot has fallen. True when pitch or roll is more than 1 rad.
        """
        # _, orientation = p.getBasePositionAndOrientation(self.robot_uid)
        # euler_orientation = p.getEulerFromQuaternion(orientation)
        # return (
        #     np.fabs(euler_orientation[0]) > 1
        #     or np.fabs(euler_orientation[1]) > 1
        #     or np.fabs(euler_orientation[2]) > 1
        # )
        
        return False
