import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

from sklearn.preprocessing import normalize


## Hyper Params
MAX_EPISODE_LEN = 50  # Number of steps for one training episode
REWARD_FACTOR = 10
BOUND_ANGLE = 30
STEP_ANGLE = 15 # Maximum angle delta per step
JOINT_LENGTH = 0.05

def leg_IK(angle, length, offset, sign=1):
    """
    Returns each angle in the joint of the leg, to match the desired swing angle and leg extension (length).
    """
    length = max(length, JOINT_LENGTH * 0.2)

    # Inner angle alpha
    cosAngle0 = (length**2) / (2 * JOINT_LENGTH * length)
    alpha = np.arccos(cosAngle0) * sign + angle
    #if alpha < 0:
    #    sign = -sign
    # Inner angle beta
    cosAngle1 = (-(length**2) + JOINT_LENGTH**2 + JOINT_LENGTH**2) / (2 * JOINT_LENGTH * JOINT_LENGTH)
    beta = -sign * (np.pi - np.arccos(cosAngle1)) + offset

    if math.isnan(alpha):
        print("alpha is nan")
        alpha = 0
    if math.isnan(beta):
        print("beta is nan")
        beta = 0

    return alpha, beta

def joint_extension(x):
    return x
    #return sqrt(x/2) * 2

class OpenCatGymEnv(gym.Env):
    """ Gym environment (stable baselines 3) for OpenCat robots.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        self.step_counter = 0
        # Store robot state and joint history
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])

        # Max joint angles
        self.boundAngles = np.deg2rad(BOUND_ANGLE)

        # Create the simulation, p.GUI for GUI, p.DIRECT for only training
        # Use options="--opengl2" if it decides to not work?
        if render:
            p.connect(p.GUI)#, options="--opengl2") #, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncomment to create a video
        else:
            p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(fixedTimeStep=1.0/60)

        # Stop rendering
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # Move camera
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=-10,
                                     cameraPitch=-40, 
                                     cameraTargetPosition=[0.4,0,0])
        
        # The action space contains the 11 joint angles
        self.action_space = spaces.Box(np.array([-1]*11), np.array([1]*11))

        # The observation space are the torso roll, pitch and the joint angles and a history of the last 20 joint angles
        # 11 * 20 + 6 = 226       
        self.observation_space = spaces.Box(np.array([-1]*226), np.array([1]*226))
 
    def get_desired_joint_angles(self, jointAngles, action):
        """"Adds the action vector to the revolute joints. Joint angles are
        clipped. `jointAngles` is changed to have the updated angles of the
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

        desiredJointAngles = jointAngles
        ds = np.deg2rad(STEP_ANGLE) # Maximum joint angle derivative (maximum change per step), should be implemented in setJointMotorControlArray
        
        # Use IK to compute the new angles of each joint
        desired_left_front_angle = np.deg2rad(BOUND_ANGLE * action[0])
        desired_left_front_length = JOINT_LENGTH * joint_extension(action[1] + 1)

        desired_right_front_angle = np.deg2rad(BOUND_ANGLE * action[2])
        desired_right_front_length = JOINT_LENGTH * joint_extension(action[3] + 1)

        desired_left_rear_angle = np.deg2rad(BOUND_ANGLE * action[4])
        desired_left_rear_length = JOINT_LENGTH * joint_extension(action[5] + 1)

        desired_right_rear_angle = np.deg2rad(BOUND_ANGLE * action[6])
        desired_right_rear_length = JOINT_LENGTH * joint_extension(action[7] + 1)

        # Compute the new angles of the joints
        desiredJointAngles[9:11] = leg_IK(desired_left_front_angle + -np.pi / 4, desired_left_front_length, np.pi/2)
        desiredJointAngles[2:4] = leg_IK(desired_right_front_angle + np.pi / 4, desired_right_front_length, -np.pi/2, -1)
        desiredJointAngles[7:9] = leg_IK(desired_left_rear_angle, desired_left_rear_length, -np.pi/2, -1)
        desiredJointAngles[0:2] = leg_IK(desired_right_rear_angle, desired_right_rear_length, np.pi/2)

        desiredJointAngles[4] += ds * action[8]
        desiredJointAngles[5] += ds * action[9]
        desiredJointAngles[6] += ds * action[10]

        # TODO: Clip the joint angles for the tail, neck and head
        desiredJointAngles[4] = np.clip(desiredJointAngles[4], -np.pi/2, np.pi/2)
        desiredJointAngles[5] = np.clip(desiredJointAngles[5], -np.pi/2, np.pi/2)
        desiredJointAngles[6] = np.clip(desiredJointAngles[6], -np.pi/2, np.pi/2)

        return desiredJointAngles


    def step(self, action):
        # `action` is a vector of values -1 <= x <= 1 .

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        lastPosition = p.getBasePositionAndOrientation(self.robotUid)[0]
        jointAngles = np.asarray(p.getJointStates(self.robotUid, self.jointIds), dtype=object)[:,0]

        #if(self.step_counter % 2 == 0): # Every 2nd iteration will be added to the joint history
        self.jointAngles_history = np.append(self.jointAngles_history, action)
        self.jointAngles_history = np.delete(self.jointAngles_history, np.s_[0:11])

        desiredJointAngles = self.get_desired_joint_angles(jointAngles, action)
          
        # Set new joint angles - the forces, positionGains and velocityGains are very important here and will likely not match the real world
        p.setJointMotorControlArray(self.robotUid, self.jointIds, p.POSITION_CONTROL, desiredJointAngles)#, forces=[6]*11, positionGains=[0.05]*11, velocityGains=[0.8]*11)
        
        # Step through the simulation 3 times to simulate 20hz input
        p.stepSimulation()
        p.stepSimulation()
        p.stepSimulation()
        
        # Read robot state (pitch, roll and their derivatives of the torso-link)
        # Get pitch and roll of torso
        state_robot_pos, state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)
        state_robot_ang_euler = np.asarray(p.getEulerFromQuaternion(state_robot_ang)[0:2])
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel_norm = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel_norm.reshape(1,-1)[0]))
        
        # Reward is the advance in x-direction - deviation in the y-direction
        currentPosition = p.getBasePositionAndOrientation(self.robotUid)[0] # Position of torso-link
        foward_factor = currentPosition[0] - lastPosition[0]
        horizontal_factor = (abs(currentPosition[1]) - abs(lastPosition[1])) / 2
        vertical_factor = 0
        if currentPosition[2] > 0.08:
            vertical_factor = abs(currentPosition[2] - lastPosition[2]) / 2
        reward = (foward_factor - horizontal_factor - vertical_factor) * REWARD_FACTOR
        done = False
        
        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        # No debug info
        info = {}
        self.observation = np.hstack((self.state_robot, self.jointAngles_history))

        return np.array(self.observation).astype(np.float32), reward, done, info
        
    def reset(self):     
        """Reset the simulation to its original state."""
        
        self.step_counter = 0
        p.resetSimulation()

        # Disable rendering during loading
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) 
        p.setGravity(0,0,-9.81)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

        planeUid = p.loadURDF("plane.urdf")
        friction = 1.4
        p.changeDynamics(planeUid, -1, lateralFriction = friction)
        
        startPos = [0,0,0.04]
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.robotUid = p.loadURDF("models/CatModel.urdf",
                                   startPos,
                                   startOrientation, 
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # Get joint IDs of robot.
        self.jointIds = []
        for j in range(p.getNumJoints(self.robotUid)):

            jointType = p.getJointInfo(self.robotUid, j)[2]
            if (jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)

        action = [0, 0.75] * 4 + [0] * 3

        desired_left_front_angle = np.deg2rad(BOUND_ANGLE * action[0])
        desired_left_front_length = JOINT_LENGTH * joint_extension(action[1] + 1)

        desired_right_front_angle = np.deg2rad(BOUND_ANGLE * action[2])
        desired_right_front_length = JOINT_LENGTH * joint_extension(action[3] + 1)

        desired_left_rear_angle = np.deg2rad(BOUND_ANGLE * action[4])
        desired_left_rear_length = JOINT_LENGTH * joint_extension(action[5] + 1)

        desired_right_rear_angle = np.deg2rad(BOUND_ANGLE * action[6])
        desired_right_rear_length = JOINT_LENGTH * joint_extension(action[7] + 1)

        resetPos = np.array([np.pi / 4, 0, -np.pi / 4, 0, 0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0])

        resetPos[9:11] = leg_IK(desired_left_front_angle, desired_left_front_length, np.pi/2)
        resetPos[2:4] = leg_IK(desired_right_front_angle, desired_right_front_length, -np.pi/2, -1)
        resetPos[7:9] = leg_IK(desired_left_rear_angle, desired_left_rear_length, -np.pi/2, -1)
        resetPos[0:2] = leg_IK(desired_right_rear_angle, desired_right_rear_length, np.pi/2, 1)

        for i, j in enumerate(self.jointIds):
            p.resetJointState(self.robotUid, j, resetPos[i])


        # Get pitch and roll of torso
        state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)[1]

        # Get angular velocity of robot torso and normalise
        state_robot_vel = np.asarray(p.getBaseVelocity(self.robotUid)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel.reshape(1,-1)[0]))
        
        # Initialize robot state history with reset position
        self.jointAngles_history = np.tile(action, 20)
        
        self.observation = np.concatenate((self.state_robot ,self.jointAngles_history))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # Re-activate rendering
        
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", when pitch or roll is more than 1 rad.
        """
        position, orientation = p.getBasePositionAndOrientation(self.robotUid)
        orientation = p.getEulerFromQuaternion(orientation)
        is_fallen = np.fabs(orientation[0]) > 1 or np.fabs(orientation[1]) > 1
        
        return is_fallen
