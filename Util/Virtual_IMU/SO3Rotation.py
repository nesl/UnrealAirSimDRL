# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:09:24 2018

@author: natsn
"""
from euclid import Vector3, Quaternion
import numpy as np
from mpl_toolkits.mplot3d import  Axes3D
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from matplotlib import cm


# Numpy arrays
def rotate_rodriguez(v,rot_axis,angle):
    rot_axis_norm = rot_axis / np.linalg.norm(rot_axis)
    v_parallel = np.dot(v,rot_axis_norm)*rot_axis_norm
    v_perp = v - v_parallel
    v_perp_prime = np.cos(angle)*v_perp + np.sin(angle)*np.cross(v,rot_axis) # could do v_parr x n...same answer...only perp components
    return v_parallel + v_perp_prime

# input degrees
def get_rotation_matrix_ypr(vector, yaw, pitch, roll, isRTransposed = False):
    phsi_r = np.deg2rad(yaw)
    theta_r = np.deg2rad(pitch)
    phi_r = np.deg2rad(roll)
    
    R_rb_phsi = np.matrix([[np.cos(np.cos(phsi_r)), -1*np.sin(phsi_r), 0],
                            [np.sin(phsi_r), np.cos(phsi_r), 0],
                            [0,0,1]])
    
    R_rb_theta = np.matrix([[np.cos(theta_r), 0, np.sin(theta_r)],
                             [0, 1, 0],
                             [-np.sin(theta_r), 0, np.cos(theta_r)]])
    
    R_rb_theta = np.matrix([[1, 0, 0],
                            [0, np.cos(phi_r), -np.sin(phi_r)],
                            [0, np.sin(phi_r), np.cos(phi_r)]])
    if isRTransposed:
        return (R_rb_phsi*R_rb_theta*R_rb_theta).T*np.matrix(vector).T
    else:
        return R_rb_phsi*R_rb_theta*R_rb_theta*np.matrix(vector).T

def get_numpy_quaternion(q):
    return np.array(q[0],q[1],q[2],q[3])

def get_quaternion(angle, rot_axis):
    angle = angle/2
    quat = np.array([np.cos(angle), np.sin(angle)*rot_axis[0], 
                            np.sin(angle)*rot_axis[1], np.sin(angle)*rot_axis[2]])
    return quat

def get_quaternion_conjugate(quat):
    quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    return quat_conj

# returns the corresponding quaternion and its conjugate needed for multiplication
def get_quaternion_and_conjugate(angle, rot_axis):
    angle = angle / 2
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    quat = get_quaternion(angle, rot_axis)
    quat_conj = get_quaternion_conjugate(quat)
    return (quat, quat_conj)

def get_angle_rot_axis(quat):
    ai = quat[1] / np.linalg.norm(quat)
    aj = quat[2] / np.linalg.norm(quat)
    ak = quat[3] / np.linalg.norm(quat)
    angle = 2*np.arctan2(np.sqrt(quat[1]**2 + quat[2]**2 + quat[3]**2), quat[0])
    return (angle, np.array([ai, aj, ak]))

# returns a 4d matrix
def get_quat_as_mat(q1):
    return np.matrix(np.array([[q1[0],-q1[1],-q1[2],-q1[3]],
                          [q1[1], q1[0], -q1[3], q1[2]],
                          [q1[2],q1[3],q1[0],-q1[1]],
                          [q1[3],-q1[2],q1[1],q1[0]]]))

# input quaternions
def quaternion_mat_vec_mul(q1, q2):
    q1 = get_quat_as_mat(q1)
    q2 = np.matrix(q2.reshape(len(q2),1)) # matrix vector
    return np.array(q1*q2)

# input quaternions
def quaternion_mat_mat_mul(q1, q2):
    q1 = get_quat_as_mat(q1)
    q2 = get_quat_as_mat(q2)
    return np.array(q1*q2)

# returns a quaternion with the yaw pitch roll convention
def euler_angles_to_quaternion(yaw, pitch, roll):
    q0 = np.cos(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.sin(pitch/2)*np.sin(yaw/2)
    q1 = np.sin(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) - np.cos(roll/2)* np.sin(pitch/2)*np.sin(yaw/2)
    q2 = np.cos(roll/2)*np.sin(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.cos(pitch/2)*np.sin(yaw/2)
    q3 = np.cos(roll/2)*np.cos(pitch/2)*np.sin(yaw/2) - np.sin(roll/2)*np.sin(pitch/2)*np.cos(yaw/2)
    quat = np.array([q0,q1,q2,q3])
    return quat

# returns a euler (yaw, pitch, roll) with yaw pitch roll convention
def quaternion_to_euler_angles(quat):
    roll = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2))
    pitch = np.arcsin(-2*(quat[1]* quat[3] - quat[0]*quat[2]))
    yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]* quat[2]),1 - 2*(quat[2]**2 + quat[3]**2))
    return (yaw, pitch, roll)     

# Form 1, same answer
def quaternion_to_rotation_matrix(quat):
    rot_mat =  np.matrix(np.array([[quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2,
               2*(quat[1]*quat[2] - quat[0]*quat[3]),
               2*(quat[0]*quat[2] + quat[1]*quat[3])],
               [2*(quat[1]*quat[2] + quat[0]*quat[3]),
               quat[0]**2 - quat[1]**2 + quat[2]**2 - quat[3]**2,
               2*(quat[2]*quat[3] - quat[0]*quat[1])],
               [2*(quat[1]*quat[3] - quat[0]*quat[2]),
               2*(quat[0]*quat[1] + quat[2]*quat[3]),
               quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2]]))
    return rot_mat

# Form 2, same answer
def quaternion_to_rotation_matrix2(quat):
    rot_mat = np.matrix(np.array([[1 - 2*(quat[2]**2 + quat[3]**2), 2*(quat[1]*quat[2] - quat[0]*quat[3]), 2*(quat[0]*quat[2] + quat[1]* quat[3])],
                                   [2*(quat[1]*quat[2] + quat[0]*quat[3]), 1 - 2*(quat[1]**2 + quat[3]**2), 2*(quat[2]*quat[3] - quat[0]*quat[1])],
                                   [2*(quat[1]*quat[3] - quat[0]*quat[2]), 2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2)]]))
    return rot_mat



# returns a quaternion 
def rotate_vector_by_quaternion(vector, quat):
# old way -- tried and true
    #tmp = quaternion_mat_vec_mul(quat, vec_quat)
#    result = quaternion_mat_vec_mul(tmp, quat_conj)
    
    vec_quat = np.concatenate((np.array([0]),vector))
    quat_conj = get_quaternion_conjugate(quat)
    
    vec_quat = np.matrix(vec_quat).reshape(len(vec_quat),1)
    vec_quat_m = get_quat_as_mat(vec_quat)
    #quat_vec = np.matrix(quat).reshape(len(quat),1)
    quat_m = get_quat_as_mat(quat)
    quat_conj_vec = np.matrix(quat_conj).reshape(len(quat_conj),1)
    #quat_conj_m = get_quat_as_mat(quat_conj)
    result = np.array(quat_m*vec_quat_m*quat_conj_vec)
    
    return np.array(result[1:4]).reshape(-1)

  
def rotate_vector_by_angle_axis(vector, rot_axis, angle):
    quat = get_quaternion(angle, rot_axis)
    rotated_vec = rotate_vector_by_quaternion(vector, quat)
    return rotated_vec



# n stands for the rotational axis unit vector
def rotate_vector_ypr_quaternion_method(vector, rot_axis_yaw, rot_axis_pitch, rot_axis_roll, yaw, pitch, roll):
    n_yaw = rot_axis_yaw / np.linalg.norm(rot_axis_yaw)
    n_pitch = rot_axis_pitch / np.linalg.norm(rot_axis_pitch)
    n_roll = rot_axis_roll / np.linalg.norm(rot_axis_roll)
    
    vector = rotate_vector_by_angle_axis(vector, n_yaw, yaw)
    n_pitch = rotate_vector_by_angle_axis(n_pitch, n_yaw, yaw)
    n_roll = rotate_vector_by_angle_axis(n_roll, n_yaw, yaw)
    
    vector = rotate_vector_by_angle_axis(vector, n_pitch, pitch)
    n_yaw = rotate_vector_by_angle_axis(n_yaw, n_pitch, pitch)
    n_roll = rotate_vector_by_angle_axis(n_roll, n_pitch, pitch)

    vector = rotate_vector_by_angle_axis(vector, n_roll, roll)
    n_yaw = rotate_vector_by_angle_axis(n_yaw, n_roll, roll)
    n_pitch = rotate_vector_by_angle_axis(n_pitch, n_roll, roll)
    return [vector, [n_yaw, n_pitch, n_roll]]


# n stands for the rotational axis unit vector
def rotate_vector_frame_ypr(vector,frame, ypr):
    
    quat = euler_angles_to_quaternion(ypr[0], ypr[1], ypr[2])
    rotated_vec = rotate_vector_by_quaternion(vector, quat) 
    frame_x = rotate_vector_by_quaternion(frame[0], quat)
    frame_y = rotate_vector_by_quaternion(frame[1], quat)
    frame_z = rotate_vector_by_quaternion(frame[2], quat)

    return [rotated_vec, [frame_x, frame_y, frame_z]]

# n stands for the rotational axis unit vector
def rotate_vector_frame_ypr_rotation_matrix(vector,frame, ypr):
    
    quat = euler_angles_to_quaternion(ypr[0], ypr[1], ypr[2])
    rot_mat = quaternion_to_rotation_matrix(quat)
    
    rotated_vec = rot_mat*np.matrix(vector).reshape(len(vector),1)
    frame_x = rot_mat*np.matrix(frame[0]).reshape(len(frame[0]),1)
    frame_y = rot_mat*np.matrix(frame[1]).reshape(len(frame[1]),1)
    frame_z = rot_mat*np.matrix(frame[2]).reshape(len(frame[2]),1)
    return [rotated_vec, [frame_x, frame_y, frame_z]]


# Does single transform at a time per axis for yaw pitch and roll. Carries other axis along
def rotate_vectors_ypr_per_axis(vectors, frame, ypr):
    frames = []
    for i in range(len(vectors)):
        vectors[i],frame = rotate_vector_ypr_quaternion_method(vectors[i],ypr[0], ypr[1], ypr[2], 
                                            frame[0], frame[1], frame[2])
        frames.append(frame)
    return vectors, frames


def generate_vector_circle(vector, rot_axis, num_vects):
    angle = np.pi*2/ num_vects
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    vectors = []
    v_new = vector
    for i in range(num_vects):
        v_new = rotate_vector_by_quaternion(v_new, rot_axis, angle)
        vectors.append(v_new)
    return vectors

# list of vectors, 3d plot axis handler
def plot_vectors(vecs, ax):
    c=np.random.rand(3,)
    for v in vecs:
        ax.scatter(v[0],v[1], v[2], c = c)
        ax.plot3D([0,v[0]],[0,v[1]], [0,v[2]], color = c)
    plt.show()

# Frame = list of unit vectors comprising the 3 coordinate axes xyz relative to absolute ground truth
def rotate_plot_vector_and_body_frame(vec, frame, ypr, ax, plot_both = True):
    vec_prime, frame_prime = rotate_vector_ypr_quaternion_method(vec, frame[0], frame[1], frame[2], ypr[0], ypr[1], ypr[2])
    if plot_both:
        plot_vectors(frame + [vec], ax)
    plot_vectors(frame_prime + [vec_prime], ax)
    return (vec_prime, frame_prime)



def propogate_rotations_ypr(vec, frame, ypr, ax, steps = 5):
    new_vec = vec
    new_axes = frame
    plot_both = True
    for i in range(steps):
        old_axes = new_axes
        old_vec = new_vec
        new_vec, new_axes = rotate_plot_vector_and_body_frame(old_vec, old_axes, ypr, ax,  plot_both = plot_both)
        plot_both = False

def propogate_vector_with_ypr_cmd(vec, frame, ypr, ax, steps = 5):
    rotated_vec = vec
    new_axes = frame
    plot_vectors([rotated_vec], ax)
    for i in range(steps):
        rotated_vec, new_axes = rotate_vector_frame_ypr(rotated_vec, new_axes, ypr)
        plot_vectors([rotated_vec] + new_axes, ax)

def propogate_vector_with_ypr_cmd_rot_mat(vec, frame, ypr, ax, steps = 5):
    rotated_vec = vec
    new_axes = frame
    plot_vectors([rotated_vec], ax)
    for i in range(steps):
        rotated_vec, new_axes = rotate_vector_frame_ypr_rotation_matrix(rotated_vec, new_axes, ypr)
        plot_vectors([rotated_vec] + new_axes, ax)
    
    

def set_up_3d_plots(fignum):     
    fig = plt.figure(fignum, figsize = (6,6))
    ax = Axes3D(fig) 
    #plt.ion()
    ax.set_aspect('equal')
    ax.set_xlim3d(-1,1) 
    ax.set_ylim3d(-1,1) 
    ax.set_zlim3d(-1,1)
    return ax

if __name__ == "__main__":
    i_hat = np.array([1,0,0])
    j_hat = np.array([0,1,0])
    k_hat = np.array([0,0,1])
    
    ihat_rot_axis = np.array([1.5,0,0])
    jhat_rot_axis = np.array([0,1.5,0])
    khat_rot_axis = np.array([0,0,1.5])
    
    ref_frame = [ihat_rot_axis, jhat_rot_axis, khat_rot_axis]
    
    steps = 6
    yaw = np.pi / steps
    pitch = 0
    roll = 0
    ypr = [yaw, pitch, roll]
    vec1 = np.array([1,1,1])
    
    ax1 = set_up_3d_plots(1)
    propogate_rotations_ypr(vec1, ref_frame, ypr, ax1, steps)
    
    ax2 = set_up_3d_plots(2)
    propogate_vector_with_ypr_cmd(vec1, ref_frame, ypr, ax2, steps)
    
    ax3 = set_up_3d_plots(3)
    propogate_vector_with_ypr_cmd_rot_mat(vec1, ref_frame, ypr, ax3, steps)



    
    
    
    
    
    
    
    