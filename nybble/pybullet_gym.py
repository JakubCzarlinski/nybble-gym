"""Create a custom gym environment for the Nybble cat robot."""
from typing import Sequence, Tuple
import time
import numpy as np
import numpy.typing as npt
import pybullet as p
import pybullet_data

from gym import Env
from gym import spaces
from sklearn.preprocessing import normalize

from .utils import controls
from .utils.reward import reward_function

## Hyper Params
MAX_EPISODE_LEN = 20  # Number of steps for one training episode
ACTION_HISTORY = 10

class PybulletGym(Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, realtime=False):
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
            
        self.realtime = realtime

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

        # The action space contains the 8 joint angles
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # The observation space are the robot velocity + orientation and a history of the last X actions
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8 * ACTION_HISTORY + 6,), dtype=np.float32)

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
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Keep track of the last 5 joint angle states. This is used as part of the observation.
        self.action_history = np.append(self.action_history, action)
        self.action_history = np.delete(self.action_history, np.s_[0:8])

        prev_pos, _ = p.getBasePositionAndOrientation(self.robot_uid)
        desired_joint_angles = controls.compute_desired_angles(action, True)

        # Set new joint angles - the forces, positionGains and velocityGains are very important here
        # and will likely not match the real world
        p.setJointMotorControlArray(
            self.robot_uid,
            self.joint_ids,
            p.POSITION_CONTROL,
            np.concatenate((desired_joint_angles[0:4], np.asarray([0, 0, 0]).astype(np.float32), desired_joint_angles[4:8])),
        ) #, forces=[6]*11, positionGains=[0.05]*11, velocityGains=[0.8]*11)

        # Step through the simulation 30 times to simulate 2hz input
        for _ in range(30):
            p.stepSimulation()
            if self.realtime:
                time.sleep(1.0/60.0)

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        # Get pitch and roll of torso
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_uid)
        self.robot_state = self.get_robot_state()

        reward = reward_function(
            np.asarray(robot_pos, dtype=np.float32),
            np.asarray(prev_pos, dtype=np.float32),
            self.robot_state,
            self.action_history,
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
            self.action_history,
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
        friction = 1.0 + np.random.rand() * 0.2
        p.changeDynamics(plane_uid, -1, lateralFriction = friction)

        robot_start_pos = [0, 0, 0.04]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_uid = p.loadURDF(
            "meshes/CatModel.urdf",
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
        ]).astype(np.float32)

        # Set initial joint angles with some random noise
        reset_pos = controls.compute_desired_angles(action, True)
        reset_pos += np.random.uniform(-np.pi / 8, np.pi / 8, 8).astype(np.float32)
        for i, j in enumerate(self.joint_ids):
            if i < 4:
                p.resetJointState(self.robot_uid, j, reset_pos[i])
            elif i > 6:
                p.resetJointState(self.robot_uid, j, reset_pos[i - 3])
            else:
                p.resetJointState(self.robot_uid, j, 0)
                

        # Initialize robot state history with reset position for X steps. This is as if the robot
        # was standing still.
        self.robot_state = self.get_robot_state()
        self.action_history = np.tile(action, ACTION_HISTORY)
        observation = np.concatenate((self.robot_state, self.action_history))

        # Re-activate rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation

    def get_robot_state(self) -> npt.NDArray[np.float32]:
        """Create a state vector of the robot."""
        # Get pitch and roll of torso
        _, robot_orientation = p.getBasePositionAndOrientation(self.robot_uid)

        # Get velocity of robot torso and normalise
        robot_vel = np.asarray(p.getBaseVelocity(self.robot_uid)[1])
        robot_vel = robot_vel[0:2] # z axis is not used
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
