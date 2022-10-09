"""Create a custom gym environment for the Nybble cat robot."""
from typing import Sequence
import numpy as np
import numpy.typing as npt
import pybullet as p
import pybullet_data

from gym import Env
from gym import spaces
from numba import jit
from sklearn.preprocessing import normalize

import kitty_controls

## Hyper Params
MAX_EPISODE_LEN = 150  # Number of steps for one training episode

@jit('float32(float32[:], float32[:], float32[:], float32[::], float32[:])', nopython=True)
def reward_function(current_pos: npt.NDArray[np.float32],
                    previous_pos: npt.NDArray[np.float32],
                    current_state: npt.NDArray[np.float32],
                    previous_angles: npt.NDArray[np.float32],
                    desired_angles: npt.NDArray[np.float32]) -> np.float32:
    """Calculate reward based on the current state of the robot."""
    weights = np.array([50, -50, -5, -50, -0.005], dtype=np.float32)

    # Reward robot for moving forward
    forward_factor = current_pos[0] - previous_pos[1]

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


class OpenCatGymEnv(Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        """ Initialize the environment."""
        # Number of time steps the environment has been running for.
        self.step_counter = 0

        # Create the simulation, p.GUI for GUI, p.DIRECT for only training
        # Use options="--opengl2" if it decides to not work?
        if render:
            p.connect(p.GUI)
            # p.connect(p.GUI, options="--opengl2")
            # p.connect(p.GUI,
            #           options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60")
        else:
            p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(fixedTimeStep=1.0/60.0)

        # Stop rendering
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Move camera
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=-10,
            cameraPitch=-40,
            cameraTargetPosition=[0.4, 0, 0],
        )

        # The action space contains the 11 joint angles
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)

        # The observation space are the torso roll, pitch and the joint angles and a history of the
        # last 20 joint angles (11 * 20 + 6 = 226)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(226,), dtype=np.float32)

        self.robot_uid: int = 0
        self.joint_ids: Sequence[int] = []
        self.joint_angles_history = np.array([], dtype=np.float32)

        # Robot state consists of the orientation and velocity of the robot
        self.robot_state: npt.NDArray[np.float32] = np.zeros(6, dtype=np.float32)

    def step(self,
             action: npt.NDArray[np.float32],
             ) -> tuple[npt.NDArray[np.float32], np.float32, bool, dict]:
        """ Perform one step in the simulation. `action` is a vector of values -1 <= x <= 1 .
        """
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Keep track of the last 20 joint angle states. This is used as part of the observation.
        self.joint_angles_history = np.append(self.joint_angles_history, action)
        self.joint_angles_history = np.delete(self.joint_angles_history, np.s_[0:11])

        prev_pos, _ = p.getBasePositionAndOrientation(self.robot_uid)
        joint_angles = np.asarray(
            p.getJointStates(self.robot_uid, self.joint_ids),
            dtype=object,
        )[:,0].astype(np.float32)
        desired_joint_angles = kitty_controls.compute_desired_angles(joint_angles, action)

        # Set new joint angles - the forces, positionGains and velocityGains are very important here
        # and will likely not match the real world
        p.setJointMotorControlArray(
            self.robot_uid,
            self.joint_ids,
            p.POSITION_CONTROL,
            desired_joint_angles,
        ) #, forces=[6]*11, positionGains=[0.05]*11, velocityGains=[0.8]*11)

        # Step through the simulation 3 times to simulate 20hz input
        p.stepSimulation()
        p.stepSimulation()
        p.stepSimulation()

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        # Get pitch and roll of torso
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_uid)
        self.robot_state = self.get_robot_state()

        reward = reward_function(
            np.asarray(robot_pos, dtype=np.float32),
            np.asarray(prev_pos, dtype=np.float32),
            self.robot_state,
            self.joint_angles_history,
            desired_joint_angles,
        )

        # Stop criteria of current learning episode: Number of steps or robot fell
        done = False
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN or self.is_fallen():
            reward += np.float32(self.step_counter - MAX_EPISODE_LEN)
            done = True

        # No debug info
        info = {}
        observation = np.concatenate((
            self.robot_state,
            self.joint_angles_history,
        ))

        return observation, reward, done, info

    def reset(self):
        """Reset the simulation to its original state."""
        # Disable rendering during loading
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.step_counter = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load Assests
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_uid = p.loadURDF("plane.urdf")
        friction = 1.2 + np.random.rand() * 0.2
        p.changeDynamics(plane_uid, -1, lateralFriction = friction)

        robot_start_pos = [0, 0, 0.04]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_uid = p.loadURDF(
            "models/CatModel.urdf",
            robot_start_pos,
            robot_start_orientation,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Get joint IDs of robot.
        self.joint_ids = []
        for j in range(p.getNumJoints(self.robot_uid)):
            joint_type = p.getJointInfo(self.robot_uid, j)[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_ids.append(j)

        action = np.array([
            0.0, 0.75,     # Angle of upper then lower left front leg
            0.0, 0.75,     # Angle of upper then lower right front leg
            0.0, 0.75,     # Angle of upper then lower left back leg
            0.0, 0.75,     # Angle of upper then lower right back leg
            0.0, 0.0, 0.0, # Head neck tail angles, not sure about order.
        ]).astype(np.float32)

        # Set initial joint angles with some random noise
        reset_pos = np.zeros(11, dtype=np.float32)
        reset_pos = kitty_controls.compute_desired_angles(reset_pos, action)
        reset_pos += np.random.uniform(-np.pi / 8, np.pi / 8, 11).astype(np.float32)
        for i, j in enumerate(self.joint_ids):
            p.resetJointState(self.robot_uid, j, reset_pos[i])

        # Initialize robot state history with reset position for 20 steps. This is as if the robot
        # was standing still for 1 second.
        self.robot_state = self.get_robot_state()
        self.joint_angles_history = np.tile(action, 20)
        observation = np.concatenate((self.robot_state, self.joint_angles_history))

        # Re-activate rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation

    def get_robot_state(self) -> npt.NDArray[np.float32]:
        """Create a state vector of the robot."""
        # Get pitch and roll of torso
        _, robot_orientation = p.getBasePositionAndOrientation(self.robot_uid)

        # Get angular velocity of robot torso and normalise
        robot_vel = np.asarray(p.getBaseVelocity(self.robot_uid)[1])
        robot_vel = robot_vel[0:2]
        robot_vel_norm = normalize(robot_vel.reshape(-1,1))

        return np.concatenate((
            robot_orientation,
            robot_vel_norm.reshape(1,-1)[0],  # type: ignore
        )).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self) -> bool:
        """ Check if robot has fallen. True when pitch or roll is more than 1 rad.
        """
        _, orientation = p.getBasePositionAndOrientation(self.robot_uid)
        euler_orientation = p.getEulerFromQuaternion(orientation)
        return (
            np.fabs(euler_orientation[0]) > 1
            or np.fabs(euler_orientation[1]) > 1
            or np.fabs(euler_orientation[2]) > 1
        )
