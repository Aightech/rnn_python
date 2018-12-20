#!/usr/bin/env python

import rospy
from std_msgs.msg import Int16,Float32,Bool,Float32MultiArray,Int16MultiArray
import rospkg 
import csv
import collections

from rnn_python import RNN_module
import scripts
import math
import numpy as np

# Read from topics values
lasers = list()
speed_left = Float32()
speed_right = Float32()

# Rate for ROS in Hz
rate = 1000
l_range = 100

#-------------------------------------------
class ExpertMixture:
    def __init__(self, nb_inputs, epsilon_g, nu_g, scaling, high, RNN_number=5):
        self.epsilon_g = epsilon_g
        self.nu_g = nu_g
        self.scaling = scaling
        
        self.s_t = collections.deque(maxlen=2)
        self.g_t = np.zeros(RNN_number)
        self.RNN_number = RNN_number
        # Create a list of RNN_number RNN_modules
        self.RNNs = list()
        for i in range(0, RNN_number):
            self.RNNs.append(RNN_module(num=i, nb_inputs=nb_inputs, scaling=scaling))

        self.pub_error = list()
        self.pub_gate = list()
        if(high):
            tmp = "high_"
        else:
            tmp = "low_"

        for number in range (1, self.RNN_number + 1):
            self.pub_error.append(rospy.Publisher('/MRE/error_' + tmp + str(number), Float32 , queue_size=10))
        for number in range (1, self.RNN_number + 1):
            self.pub_gate.append(rospy.Publisher('/MRE/gate_' + tmp + str(number), Float32 , queue_size=10))

    #-------------------------------------------
    # Target et output contiennent les valeurs obtenues du robot/simulation et celles de sortie du RNN respectivement
    # Rangees dans des dictionnaires (sous la forme dict[k-ieme RNN][j-ieme composante de l'etat])
    def norm_2_i(self, index):
        norm = 0
        for i in range(0, len(self.error)):
            norm += (self.error[index][i])**2
        return norm
    #-------------------------------------------
    def get_s_t(self):
        return self.s_t

    #-------------------------------------------
    def set_s_t(self, s):
        self.s_t = s

    #-------------------------------------------
    def get_g_t(self):
        return self.g_t

    #-------------------------------------------
    def set_g_t(self, g):
        self.g_t = g

    #-------------------------------------------
    def compute_gt_i(self, index):
        sum = 0
        for i in range(0, len(self.s_t[1])):
            sum += math.exp(self.s_t[1][i])
        if sum == 0:
            sum = 0.01
        return (math.exp(self.s_t[1][index]) / sum)

    #-------------------------------------------
    def compute_post_proba(self, index):
        sum = 0
        for i in range(0, self.RNN_number):
            sum += self.g_t[i] * math.exp((-1/(2*self.scaling**2))*self.norm_2_i(i))
        if sum == 0:
            sum = 0.01
        return self.g_t[index] * math.exp((-1/(2*self.scaling**2))*self.norm_2_i(index)) / sum

    #-------------------------------------------
    def compute_partial_L_s_k(self, index):
        return (self.compute_post_proba(index) - self.g_t[index])#(1 - self.g_t[index]) )

    #-------------------------------------------
    def compute_delta_s_k_i(self, index):
        return (self.epsilon_g * self.compute_partial_L_s_k(index) - self.nu_g * (self.s_t[1][index] - self.s_t[0][index]))

    #-------------------------------------------
    def list_sum(self, a, b):
        if (len(a) != len(b)):
            print "Error in list_sum: lists are not the same size"
            return []
        ret = []
        for i in range(0, len(a)):
            ret.append(a[i] + b[i])
        return ret
    #-------------------------------------------
    def routine_offline(self, data):
        self.data = data
        self.error = np.zeros((self.RNN_number, len(data[0])))

        tmp = [0 for i in range(0,self.RNN_number)]
        self.s_t.append(tmp)
        for t in range(0, len(self.data) - 1):
            for i in range(0, self.RNN_number):
                o = self.RNNs[i].predict(self.data[t])
                self.error[i] = np.subtract(np.array(self.data[t]),o)
                # Publish the error values
                self.pub_error[i].publish(self.norm_2_i(i))
                tmp[i] = self.RNNs[i].get_gate_state()

            self.s_t.append(tmp)

            for i in range(0, self.RNN_number):
                self.g_t[i] = self.compute_gt_i(i)
                # Publish the gating values
                self.pub_gate[i].publish(self.g_t[i])

            for i in range(0, self.RNN_number):
                self.s_t[1][i] = self.s_t[1][i] + self.compute_delta_s_k_i(i)
                self.RNNs[i].update_gate(self.g_t[i], self.s_t[1][i])
                self.RNNs[i].update_weights()

    #-------------------------------------------
    def routine_online(self, data):
        self.data = data
        self.error = np.zeros((self.RNN_number, len(data)))

        tmp = [0 for i in range(0,self.RNN_number)]
        self.s_t.append(tmp)
        for i in range(0, self.RNN_number):
            o = self.RNNs[i].predict(self.data)
            self.error[i] = np.subtract(np.array(self.data),o)
            # Publish the error values
            self.pub_error[i].publish(self.norm_2_i(i))
            tmp[i] = self.RNNs[i].get_gate_state()

        self.s_t.append(tmp)

        for i in range(0, self.RNN_number):
            self.g_t[i] = self.compute_gt_i(i)

            # Publish the gating values
            self.pub_gate[i].publish(self.g_t[i])

        for i in range(0, self.RNN_number):
            self.s_t[1][i] = self.s_t[1][i] + self.compute_delta_s_k_i(i)
            self.RNNs[i].update_gate(self.g_t[i], self.s_t[1][i])
            # self.RNNs[i].update_weights()

#-------------------------------------------
def callback_lasers(data):
    global lasers
    lasers=list(data.data)
    for i in range(0, len(lasers)):
        if(lasers[i] == -1):
            lasers[i] = l_range

#-------------------------------------------
def callback_speed_left(data):
    global speed_left
    speed_left = data

#-------------------------------------------
def callback_speed_right(data):
    global speed_right
    speed_right = data

#-------------------------------------------
# This function reads csv files containing data on motors and sensors generated by a rosbag
def read_csv(path_to_file, dict_to_fill, list_flag):
    with open(path_to_file, 'rb') as csvfile:
        next(csvfile)
        r = csv.reader(csvfile, delimiter=',')
        k = 0
        for row in r:
            # Remove all the unwanted characters and delimit the data with ','
            a = (((row[-1].replace("[", "")).replace(" ", "")).replace("]", "")).split(',')
            if(list_flag):                                  # If we want a dict of list as an output 
                b = [float(i) for i in a]                   # Transform each element of the list from str to float
                c = [100.0 if x == -1 else x for x in b]    # Change all the unknown lasers value from -1 to lasers range
                c = c[1:7]                                  # Only use the 6 side and front captors
            else:                                           # If we want a dict of float as an output
                c = float(a[0])                             # Convert the first element of the list from str to float
            # dict_to_fill[float(row[0])] = c                 # Return a dict with the rostime as a key
            dict_to_fill[k] = [float(row[0]), c]                 # Return a dict with the time step as key and [rostime, data] as value
            k += 1
   
#-------------------------------------------
# This function associates the readings from the lasers which are sampled at a higher rate than the motors 
# to the corresponding readings from the motors and we normalize these values
def align_data(m_left, m_right, lasers):
    k = 0
    data = []
    for i in range(0, len(lasers)):
        if(m_left[k][0] < lasers[i][0] and k != len(m_left) - 1):
            k += 1
        data.append([m_left[k][1], m_right[k][1]] + [(a-50)/50 for a in lasers[i][1]])
    return data

#-------------------------------------------
def data_to_window(data, time, window_size):
    if(time < window_size - 1 and time > 0):
        return data[0:(time + 2)]           # Return everything we can according to current time
    elif(time > 0):
        return data[(time - window_size +2):time + 2]   # Return a window vector of [t-2, t-1, t, t+1] according to current time
    print "Error in data_to_window: check time input"

#-------------------------------------------
def extend_data(data, n):
    if(n == 0):
        return data
    return data + extend_data(data, n - 1)

#-------------------------------------------
def online_learning():
    rospy.init_node('online_learning', anonymous=True)

    # The node receives sensory and motor information from simu_fastsim:
    rospy.Subscriber("/simu_fastsim/lasers", Float32MultiArray, callback_lasers)
    rospy.Subscriber('/simu_fastsim/speed_left', Float32, callback_speed_left)
    rospy.Subscriber('/simu_fastsim/speed_right', Float32, callback_speed_right)

    # Targetted operating frequency of the node:
    r = rospy.Rate(rate) # rate in Hz

    ##########      Import the data

    rospack = rospkg.RosPack()

    # Get the file path for rospy_tutorials
    path = rospack.get_path('rnn_python')

    # Read left motor data from csv file
    m_left_data = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_speed_left.csv', m_left_data, list_flag=False)

    # Read right motor data from csv file
    m_right_data = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_speed_right.csv', m_right_data, list_flag=False)

    # Read lasers data from csv file
    lasers_data = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_lasers.csv', lasers_data, list_flag=True)

    # Align the data
    training_data = align_data(m_left_data, m_right_data, lasers_data)

    # Extend the data
    # training_data = extend_data(training_data, 5)

    # Start time and timing related things
    startT = rospy.get_time()
    rospy.loginfo("Start time: " + str(startT))

    ##########      Learn

    offline = True;

    mre_low_off = ExpertMixture(8, 0.007, 0.02, 0.1, high=False)
    mre_high_off = ExpertMixture(5, 0.007, 0.02, 0.1, high=True)

    delta = 100
    t = 0

    if(offline):
        for d in training_data:
            mre_low_off.routine_online(d)

            if(t % delta == 0):
                g = mre_low_off.get_g_t()
                mre_high_off.routine_online(g)
            t += 1

        # mre_low_off.routine_offline(training_data)
    mre_low_on = ExpertMixture(8, 0.007, 0.02, 1, high=False)
    mre_high_on = ExpertMixture(5, 0.007, 0.02, 1, high=True)

    while (not rospy.is_shutdown() and not offline):
        data = []
        data += [speed_left.data]
        data += [speed_right.data]
        tmp = [(a-50)/50 for a in lasers[1:(len(lasers)-1)]]
        data += tmp
        if(len(data) == 8):
            mre_low_on.routine_online(data)
            if(t%delta == 0):
                g = mre_low_on.get_g_t()
                mre_high_on.routine_online(g)
            t += 1

#-------------------------------------------
if __name__ == '__main__':
    try:
        online_learning()
    except rospy.ROSInterruptException: pass