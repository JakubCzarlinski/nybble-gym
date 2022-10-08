"""Create a custom gym environment for the Nybble cat robot."""
import math
import numpy as np
import pybullet as p
import pybullet_data

from gym import Env
from gym import spaces
from sklearn.preprocessing import normalize

## Hyper Params
MAX_EPISODE_LEN = 150  # Number of steps for one training episode
REWARD_FACTOR = 10
BOUND_ANGLE = 45
STEP_ANGLE = 15 # Maximum angle delta per step
JOINT_LENGTH = 0.05
JOINT_LENGTH_SQUARED = JOINT_LENGTH**2

def leg_ik(angle, length, offset, sign=1):
    """
    Returns each angle in the joint of the leg, to match the desired swing angle
    and leg extension (length). Uses the formula: a^2 = b^2 + c^2 - 2abcos(C)
    """
    # Prevent negatives angles and angles of small magnitudes
    length = max(length, JOINT_LENGTH * 0.2)
    length_squared = length**2

    # Inner angle alpha
    upper_cos = (length_squared)/(2*JOINT_LENGTH*length)
    upper_joint = np.arccos(upper_cos)*sign + angle

    # Inner angle beta
    lower_cos = (2 * JOINT_LENGTH_SQUARED - length_squared) / (2 * JOINT_LENGTH_SQUARED)
    lower_joint = sign*(np.arccos(lower_cos) - np.pi) + offset

    if math.isnan(upper_joint):
        print("Upper joint angle is nan")
        upper_joint = 0
    if math.isnan(lower_joint):
        print("Lower joint angle is nan")
        lower_joint = 0

    return upper_joint, lower_joint


def joint_extension(angle):
    """Returns the length of the leg joint, given the angle of the joint."""
    return 2 - ((1 - angle) / 2)**2


class OpenCatGymEnv(Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        """ Initialize the environment."""
        # Number of time steps the environment has been running for.
        self.step_counter = 0

        # Store robot state and joint history.
        self.state_robot_history = np.array([])
        self.joint_angles_history = np.array([])

        # Max joint angles.
        self.bound_angles = np.deg2rad(BOUND_ANGLE)

        # Create the simulation, p.GUI for GUI, p.DIRECT for only training
        # Use options="--opengl2" if it decides to not work?
        if render:
            # Normal GUI
            p.connect(p.GUI)
            # Legacy
            # p.connect(p.GUI, options="--opengl2")
            # Video rendering
            # p.connect(p.GUI,
            #           options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60")
        else:
            p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(fixedTimeStep=1.0/60.0)

        # Stop rendering
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Move camera
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=-10,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.4, 0, 0],
        )

        # The action space contains the 11 joint angles
        self.action_space = spaces.Box(np.array([-1]*11), np.array([1]*11))

        # The observation space are the torso roll, pitch and the joint angles and a history of the
        # last 20 joint angles
        # 11 * 20 + 6 = 226
        self.observation_space = spaces.Box(np.array([-1]*226), np.array([1]*226))

    def get_desired_joint_angles(self, joint_angles, action):
        """"Adds the action vector to the revolute joints. Joint angles are
        clipped. `joint_angles` is changed to have the updated angles of the
        entire robot. The vector returned contains the only the revolute joints
        of the robot."""

        # Below is the mapping from the old 3D model to the new 3D model.
        # -----------------------------------------------------------
        # | PREVIOUS        |    NEW                   | NEW INDEX  |
        # |=========================================================|
        # | hip_right       |  R_Rear_Hip_Servo_Thigh  |  i = 1     |
        # | knee_right      |  R_Rear_Knee_Servo       |  i = 3     |
        # |---------------------------------------------------------|
        # | shoulder_right  |  R_Front_Hip_Servo_Thigh |  i = 7     |
        # | elbow_right     |  R_Front_Knee_Servo      |  i = 9     |
        # |---------------------------------------------------------|
        # | #############   |  Body_Neck_Servo         |  i = 13    |
        # | #############   |  Neck_Head_Servo         |  i = 14    |
        # | --------------------------------------------------------|
        # | #############   |  Tail_Servo_Tail         |  i = 19    |
        # |---------------------------------------------------------|
        # | hip_left        |  L_Rear_Hip_Servo_Thigh  |  i = 21    |
        # | knee_left       |  L_Rear_Knee_Servo       |  i = 23    |
        # |---------------------------------------------------------|
        # | shoulder_left   |  L_Front_Hip_Servo_Thigh |  i = 27    |
        # | elbow_left      |  L_Front_Knee_Servo      |  i = 29    |
        # -----------------------------------------------------------

        desired_joint_angles = joint_angles

        # Use IK to compute the new angles of each joint
        desired_left_front_angle = np.deg2rad(BOUND_ANGLE * action[0])
        desired_left_front_length = JOINT_LENGTH * joint_extension(action[1])

        desired_right_front_angle = np.deg2rad(BOUND_ANGLE * action[2])
        desired_right_front_length = JOINT_LENGTH * joint_extension(action[3])

        desired_left_rear_angle = np.deg2rad(BOUND_ANGLE * action[4])
        desired_left_rear_length = JOINT_LENGTH * joint_extension(action[5])

        desired_right_rear_angle = np.deg2rad(BOUND_ANGLE * action[6])
        desired_right_rear_length = JOINT_LENGTH * joint_extension(action[7])

        # Compute the new angles of the joints
        desired_joint_angles[9:11] = leg_ik(
            angle=(desired_left_front_angle - np.pi/8),
            length=desired_left_front_length,
            offset=np.pi/2,
        )
        desired_joint_angles[2:4] = leg_ik(
            angle=(desired_right_front_angle + np.pi/8),
            length=desired_right_front_length,
            offset=0,
            sign=-1,
        )
        desired_joint_angles[7:9] = leg_ik(
            angle=desired_left_rear_angle,
            length=desired_left_rear_length,
            offset=-np.pi/2,
            sign=-1,
        )
        desired_joint_angles[0:2] = leg_ik(
            angle=desired_right_rear_angle,
            length=desired_right_rear_length,
            offset=np.pi/2,
        )

        desired_joint_angles[4:6] += np.deg2rad(STEP_ANGLE) * action[8:10]
        desired_joint_angles[4:6] = np.clip(desired_joint_angles[4:6], -np.pi/2, np.pi/2)

        return desired_joint_angles

    def step(self, action):
        # `action` is a vector of values -1 <= x <= 1 .
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        last_position = p.getBasePositionAndOrientation(self.robot_uid)[0]
        joint_angles = np.asarray(
            p.getJointStates(self.robot_uid, self.joint_ids),
            dtype=object,
        )[:,0]

        #if(self.step_counter % 2 == 0): # Every 2nd iteration will be added to the joint history
        self.joint_angles_history = np.append(self.joint_angles_history, action)
        self.joint_angles_history = np.delete(self.joint_angles_history, np.s_[0:11])

        desired_joint_angles = self.get_desired_joint_angles(joint_angles, action)

        # Set new joint angles - the forces, positionGains and velocityGains are very important here
        # and will likely not match the real world
        p.setJointMotorControlArray(self.robot_uid,
                                    self.joint_ids,
                                    p.POSITION_CONTROL,
                                    desired_joint_angles,
        )
        #, forces=[6]*11, positionGains=[0.05]*11, velocityGains=[0.8]*11)

        desired_left_front_angle = np.deg2rad(BOUND_ANGLE * action[0])
        desired_right_front_angle = np.deg2rad(BOUND_ANGLE * action[2])

        # Step through the simulation 3 times to simulate 20hz input
        p.stepSimulation()
        p.stepSimulation()
        p.stepSimulation()

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        # Get pitch and roll of torso
        robot_pos, robot_orientation = p.getBasePositionAndOrientation(self.robot_uid)
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robot_uid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel_norm = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((
            robot_orientation,
            state_robot_vel_norm.reshape(1,-1)[0],
        ))

        # Reward is the advance in x-direction - deviation in the y-direction
        forward_factor = robot_pos[0] - last_position[0]
        horizontal_factor = (abs(robot_pos[1]) - abs(last_position[1])) / 2
        vertical_factor = 0
        if robot_pos[2] > 0.08:
            vertical_factor = abs(robot_pos[2] - last_position[2]) / 2

        angle_factor = abs(sum(
            (desired_joint_angles - self.joint_angles_history[0])
            - (self.joint_angles_history[0] - self.joint_angles_history[1])
        ))
        front_legs_factor = max(-0.3 - desired_left_front_angle, 0)
        front_legs_factor += max(-0.3 - desired_right_front_angle, 0)
        orientation_factor = abs(np.fabs(robot_orientation[0]))

        reward = self.reward_function(
            forward_factor,
            angle_factor,
            orientation_factor,
            front_legs_factor,
            horizontal_factor,
            vertical_factor,
        )

        # Stop criteria of current learning episode: Number of steps or robot fell
        done = False
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = (self.step_counter - MAX_EPISODE_LEN) / (MAX_EPISODE_LEN / 5)
            done = True

        # No debug info
        info = {}
        self.observation = np.hstack((self.state_robot, self.joint_angles_history))

        return np.array(self.observation).astype(np.float32), reward, done, info

    def reward_function(self,
                        forward_factor,
                        angle_factor,
                        orientation_factor,
                        front_legs_factor,
                        horizontal_factor,
                        vertical_factor):
        """Calculate reward based on the current state of the robot."""
        weights = np.array([100, -0.0025, -0.05, -0, -100, -100])
        factors = np.array([
            forward_factor,
            angle_factor,
            orientation_factor,
            front_legs_factor,
            horizontal_factor,
            vertical_factor,
        ])
        return np.dot(weights, factors)

    def reset(self):
        """Reset the simulation to its original state."""

        self.step_counter = 0
        p.resetSimulation()

        # Disable rendering during loading
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.81)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        plane_uid = p.loadURDF("plane.urdf")
        friction = 1.4
        p.changeDynamics(plane_uid, -1, lateralFriction = friction)

        start_pos = [0,0,0.04]
        start_orientation = p.getQuaternionFromEuler([0,0,0])

        self.robot_uid = p.loadURDF("models/CatModel.urdf",
                                   start_pos,
                                   start_orientation,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Get joint IDs of robot.
        self.joint_ids = []
        for j in range(p.getNumJoints(self.robot_uid)):
            joint_type = p.getJointInfo(self.robot_uid, j)[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_ids.append(j)

        action = [0, 0.75] * 4 + [0] * 3

        desired_left_front_angle = np.deg2rad(BOUND_ANGLE * action[0])
        desired_left_front_length = JOINT_LENGTH * joint_extension(action[1])

        desired_right_front_angle = np.deg2rad(BOUND_ANGLE * action[2])
        desired_right_front_length = JOINT_LENGTH * joint_extension(action[3])

        desired_left_rear_angle = np.deg2rad(BOUND_ANGLE * action[4])
        desired_left_rear_length = JOINT_LENGTH * joint_extension(action[5])

        desired_right_rear_angle = np.deg2rad(BOUND_ANGLE * action[6])
        desired_right_rear_length = JOINT_LENGTH * joint_extension(action[7])

        reset_pos = np.array([np.pi / 4, 0, -np.pi / 4, 0, 0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0])

        reset_pos[9:11] = leg_ik(desired_left_front_angle, desired_left_front_length, np.pi/2)
        reset_pos[2:4] = leg_ik(desired_right_front_angle, desired_right_front_length, 0, -1)
        reset_pos[7:9] = leg_ik(desired_left_rear_angle, desired_left_rear_length, -np.pi/2, -1)
        reset_pos[0:2] = leg_ik(desired_right_rear_angle, desired_right_rear_length, np.pi/2)

        for i, j in enumerate(self.joint_ids):
            p.resetJointState(self.robot_uid, j, reset_pos[i])


        # Get pitch and roll of torso
        state_robot_ang = p.getBasePositionAndOrientation(self.robot_uid)[1]

        # Get angular velocity of robot torso and normalise
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robot_uid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel.reshape(1,-1)[0]))

        # Initialize robot state history with reset position
        self.joint_angles_history = np.tile(action, 20)

        self.observation = np.concatenate((self.state_robot ,self.joint_angles_history))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # Re-activate rendering

        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self):
        """ Check if robot has fallen. It becomes "True", when pitch or roll is more than 1 rad.
        """
        _, orientation = p.getBasePositionAndOrientation(self.robot_uid)
        euler_orientation = p.getEulerFromQuaternion(orientation)

        return np.fabs(euler_orientation[0]) > 1 or np.fabs(euler_orientation[1]) > 1
