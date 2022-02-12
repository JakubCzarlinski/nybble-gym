import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

from sklearn.preprocessing import normalize

MAX_EPISODE_LEN = 500  # Number of steps for one training episode
REWARD_FACTOR = 10
REWARD_WEIGHT_1 = 1.0
BOUND_ANGLE = 40
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

    def __init__(self, render=True):
        self.step_counter = 0
        self.state_robot_history = np.array([])
        self.jointAngles_history = np.array([])
        self.boundAngles = np.deg2rad(BOUND_ANGLE)

        if render:
            p.connect(p.GUI) #, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncommend to create a video
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        
        # The action space are the 4 leg angles and their extensions   
        self.action_space = spaces.Box(np.array([-1]*8), np.array([1]*8))

        # The observation space are the torso roll, pitch and the joint angles and a history of the last 20 joint angles
        self.observation_space = spaces.Box(np.array([-1]*166), np.array([1]*166))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        lastPosition = p.getBasePositionAndOrientation(self.robotUid)[0]
        desiredJointAngles = np.array(action)
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
        desiredJointAngles[0:2] = leg_IK(desired_left_front_angle, desired_left_front_length, np.pi/2)
        desiredJointAngles[2:4] = leg_IK(desired_right_front_angle, desired_right_front_length, np.pi/2)
        desiredJointAngles[4:6] = leg_IK(desired_left_rear_angle, desired_left_rear_length, -np.pi/2, -1)
        desiredJointAngles[6:8] = leg_IK(desired_right_rear_angle, desired_right_rear_length, -np.pi/2, -1)

        if(self.step_counter % 2 == 0): # Every 2nd iteration will be added to the joint history
            self.jointAngles_history = np.append(self.jointAngles_history, action)
            self.jointAngles_history = np.delete(self.jointAngles_history, np.s_[0:8])

        #desiredJointAngles = [0] * 8
          
        # Set new joint angles - the forces, positionGains and velocityGains are very important here and will likely not match the real world
        p.setJointMotorControlArray(self.robotUid, self.jointIds, p.POSITION_CONTROL, targetPositions=desiredJointAngles, forces=[0.15]*8, positionGains=[0.015]*8, velocityGains=[0.2]*8)
        p.stepSimulation()
        
       # Read robot state (pitch, roll and their derivatives of the torso-link)
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
        reward = REWARD_WEIGHT_1 * (foward_factor - horizontal_factor - vertical_factor) * REWARD_FACTOR
        done = False
        
        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        info = {}
        self.observation = np.hstack((self.state_robot, self.jointAngles_history))

        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # Disable rendering during loading
        p.setGravity(0,0,-9.81)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        planeUid = p.loadURDF("plane.urdf")
        p.changeDynamics(planeUid, -1, lateralFriction = 1.4)
        
        startPos = [0,0,0.06]
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.robotUid = p.loadURDF("models/nybble.urdf",startPos, startOrientation)
        
        self.jointIds = []
        paramIds = []

        for j in range(p.getNumJoints(self.robotUid)):
            info = p.getJointInfo(self.robotUid, j)
            #print(info)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8")))

        action = [0, 0.75] * 4
        desiredJointAngles = action

        desired_left_front_angle = np.deg2rad(BOUND_ANGLE * action[0])
        desired_left_front_length = JOINT_LENGTH * joint_extension(action[1] + 1)

        desired_right_front_angle = np.deg2rad(BOUND_ANGLE * action[2])
        desired_right_front_length = JOINT_LENGTH * joint_extension(action[3] + 1)

        desired_left_rear_angle = np.deg2rad(BOUND_ANGLE * action[4])
        desired_left_rear_length = JOINT_LENGTH * joint_extension(action[5] + 1)

        desired_right_rear_angle = np.deg2rad(BOUND_ANGLE * action[6])
        desired_right_rear_length = JOINT_LENGTH * joint_extension(action[7] + 1)

        desiredJointAngles[0:2] = leg_IK(desired_left_front_angle, desired_left_front_length, np.pi/2)
        desiredJointAngles[2:4] = leg_IK(desired_right_front_angle, desired_right_front_length, np.pi/2)
        desiredJointAngles[4:6] = leg_IK(desired_left_rear_angle, desired_left_rear_length, -np.pi/2, -1)
        desiredJointAngles[6:8] = leg_IK(desired_right_rear_angle, desired_right_rear_length, -np.pi/2, -1)
    
        # Reset joint position to rest pose
        for i in range(len(paramIds)):
            p.resetJointState(self.robotUid, i, desiredJointAngles[i])

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_ang = p.getBasePositionAndOrientation(self.robotUid)[1]
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