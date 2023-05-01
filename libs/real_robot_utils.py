
import math
import os
import re
import time
from os.path import join

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

from cooperative_transport.gym_table.envs.custom_rewards import \
    custom_reward_function
from cooperative_transport.gym_table.envs.utils import WINDOW_H, WINDOW_W

# -------------------------------------------- INIT GLOBAL REAL WORLD PARAMS -------------------------------------------- #
odom_1 = None
odom_2 = None
robot_name_1 = "locobot1"
robot_name_2 = "locobot2"
if not rospy.has_param("/" + robot_name_1 + "/use_base") or rospy.get_param("/" + robot_name_1 + "/use_base") == False:
    print("Error: robot name not found...")
    exit(0)
if not rospy.has_param("/" + robot_name_2 + "/use_base") or rospy.get_param("/" + robot_name_2 + "/use_base") == False:
    print("Error: robot name not found...")
    exit(0)

# REAL WORLD PARAMETERS
# For locobot point P control i.e. distance from robot control point to center:
L_real = 0.13 # meters
r1x = -L_real
r1y = 0
r2x = L_real
r2y = 0
# For map (all in meters) ## TODO: update these values
MIN_X = -1.55 # -1.9
MAX_X = 2.1 #1.9
MAG_X = abs(MAX_X-MIN_X)
MIN_Y = 1.78 #-1.15 #-1.2
MAX_Y = -1.25 #1.9 #1.0
MAG_Y = abs(MAX_Y-MIN_Y)
sim2real_x_scale = max(MAG_X / WINDOW_W, 0.03)
sim2real_y_scale = max(MAG_Y / WINDOW_H, 0.05)
floor_scale = 20
MAX_X_VEL = np.float(MAG_Y / floor_scale)
MAX_Y_VEL = MAG_Y / floor_scale
MAX_ANG_VEL = 0.1


def move(v_x=0, v_y=0, yaw=0, duration=1.0, curr_pub=None):
    """ Move robot for a certain duration by publishing velocity command
    v_x, v_y, yaw: desired velocity and yaw
    duration: duration of movement
    curr_pub: publisher to publish velocity command
    """
    time_start = rospy.get_time()
    r = rospy.Rate(30)
    while (rospy.get_time() < (time_start + duration)):
        curr_pub.publish(Twist(linear=Vector3(x=v_x, y=v_y), angular=Vector3(z=yaw)))
        r.sleep()
    curr_pub.publish(Twist())

def share_control(f1, f2, factor=0.1):
    """ Return commanded desired table velocity given two forces applied on it
    f1, f2: 2D np.array or np.mat, each (2, 1)
    Return: desired table velocity (v_des,x v_des,y omega_z of table) np.array (3, 1)
    """
    def get_wrench(f1, f2):
        """ Return wrench vector given two forces
        f1, f2: 2D np.array or np.mat, each (2, 1)
        Return: Wrench (total F_x, F_y, T_z applied on table) np.array (3, 1)
        """
        f1_mat = np.array([
            [1, 0],
            [0, 1],
            [-r1y, r1x]
        ])
        f2_mat = np.array([
            [1, 0],
            [0, 1],
            [-r2y, r2x]
        ])

        f1 = f1.reshape(2,1)
        f2 = f2.reshape(2,1)

        return np.array((f1_mat @ f1 + f2_mat @ f2).flatten())
    des_wrench = get_wrench(f1, f2) # get sim wrench
    des_x_vel = MAX_X_VEL if (min(MAX_X_VEL, np.abs(des_wrench[0] * sim2real_x_scale)) >= MAX_X_VEL) else (des_wrench[0] * sim2real_x_scale)
    des_y_vel = MAX_Y_VEL if min(MAX_Y_VEL, np.abs(des_wrench[1] * sim2real_y_scale)) >= MAX_Y_VEL else des_wrench[1] * sim2real_y_scale
    des_ang_vel = MAX_ANG_VEL if min(MAX_ANG_VEL, np.abs(des_wrench[2])) == MAX_ANG_VEL else des_wrench[2]
    des_vel = np.array([des_x_vel, des_y_vel, des_ang_vel])
    return des_vel

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radian

def reset_odom():
    reset_odom_1 = rospy.Publisher("/" + robot_name_1 + "/mobile_base/commands/reset_odometry", Empty, queue_size=10)
    reset_odom_2 = rospy.Publisher("/" + robot_name_2 + "/mobile_base/commands/reset_odometry", Empty, queue_size=10)
    # reset odometry (these messages take a few iterations to get through)
    timer = time.time()
    while time.time() - timer < 0.25:
        reset_odom_1.publish(Empty())
        reset_odom_2.publish(Empty())

def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def make_video(dir, name):
    """Make video from images in a folder.

    Args:
        dir (str): Path to folder containing images.
        name (str): Name of video to save.
    """
    image_folder = dir
    video_name = join(dir, f"{name}.avi")

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    sorted_images = sorted(images, key=natural_sort_key)
    frame = cv2.imread(os.path.join(image_folder, sorted_images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in sorted_images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def compute_reward(
    states,
    goal,
    obs,
    env=None,
    vectorized=False,
    interaction_forces=None,
    u_r=None,
    u_h=None,
    collision=None,
    collision_checking_env=None,
    success=None,
) -> float:
    """
    Compute reward for the given state and goal.

    Args:
        states (np.ndarray): shape (N, obs_dim). Evaluating the reward for each state
                in the batch of size N.
        goal (np.ndarray): shape (2, ). Goal position
        obs (np.ndarray): shape (num_obs, 2). Obstacles positions for each obstacle
        include_interaction_forces_in_reward (bool): If True, include the interaction forces in the reward
        interaction_forces (float): If provided, use the interaction forces computed as a part of the reward fn
        vectorized (bool): Whether to vectorize the reward computation.
                In inference, this should be True since we want to sample from the model.
    """
    if env.include_interaction_forces_in_rewards:
        reward = custom_reward_function(
            states,
            goal,
            obs,
            interaction_forces=interaction_forces,
            vectorized=True,
            collision_checking_env=collision_checking_env,
            env=env,
            u_h=u_h,
        )
    else:
        reward = custom_reward_function(
            states, goal, obs, vectorized=True, env=env, collision_checking_env=collision_checking_env, u_h=u_h
        )
    return reward