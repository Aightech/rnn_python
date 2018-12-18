import rospy
from std_msgs.msg import Int16,Float32,Bool,Float32MultiArray,Int16MultiArray
import rospkg 
import csv
import collections

import math
import numpy as np

# Read from topics values
lasers = list()
speed_left = 0
speed_right = 0

# Rate for ROS in Hz
rate = 1000


T = 30
nb_sensor = 6
nb_motor = 2
nb_rnn = 4

U = np.zeros((nb_rnn, nb_sensor+nb_motor)) 
V = np.zeros((nb_sensor+nb_motor, nb_rnn))
W = np.zeros((nb_rnn, nb_rnn))

Xpredict = np.zeros((nb_sensor+nb_motor,T))
Xpredict[0,0] = 1 
Xreal = np.zeros((nb_sensor+nb_motor,T))
Xreal[0,0] = 1 

class RNN_module:
    
    def __init__(self, num, nb_sensor, nb_motor, scaling, learning_rate=0.9, hidden_dim=4, bptt_truncate=4):
        # Assign instance variables
        self.num = num
        self.gate_state = 0
        self.gate_opening = 0
        self.nb_sensor = nb_sensor
        self.nb_motor = nb_motor
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate
        # Window variables
        self.window = collections.deque(maxlen = bptt_truncate + 1)
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./(nb_sensor+nb_motor)), np.sqrt(1./(nb_sensor + nb_motor)), (hidden_dim, nb_sensor + nb_motor))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (nb_sensor + nb_motor, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.S = np.zeros( (hidden_dim, bptt_truncate + 1) )
        self.O = np.zeros( (nb_motor+nb_sensor, bptt_truncate + 1))

    #-------------------------------------------
    # x: input of the RNN (sensori-motor state at time t)
    # time: current time step
    def predict(self, x):
    	self.append_value_window(x)
    	self.O = self.forward_propagation()
    	return self.get_output()

    #-------------------------------------------
    # Updates attributes of RNN class
    def update_attributes(self, gate_opening, gate_state):
    	set_gate_opening(gate_opening)
    	set_gate_state(gate_state)

    #-------------------------------------------
    def set_gate_state(self, s):
        self.gate_state = s

    #-------------------------------------------
    def get_gate_state(self):
        return self.gate_state

    #-------------------------------------------
    def set_gate_opening(self, g):
        self.gate_opening = g

    #-------------------------------------------
    def get_gate_opening(self):
        return self.gate_opening

    #-------------------------------------------
    def append_value_window(self, x):
        self.window.append(x)

    #-------------------------------------------
    def get_window_value(self):
        return self.window

    #-------------------------------------------
    def get_output(self):
    	if(len(self.O[0]) > 0):
        	return self.O[:,-1]
        else:
        	return np.zeros(self.nb_motor+self.nb_sensor)

    #-------------------------------------------
    def forward_propagation(self):
        # The total number of time steps (we omit t + 1)
        T = len(self.window) - 1
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        self.S[:,-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((self.nb_sensor + self.nb_motor, T))
        # For each time step...
        for t in range(0, T):
            self.S[:,t] = np.tanh(self.U.dot(self.window[t][:]) + self.W.dot(self.S[:,t-1]))
            output[:,t] = self.V.dot(self.S[:,t])
        return output	# Return output on the current time window

    #-------------------------------------------
    def compute_partial_E_V(self):
    	sum = 0
    	for t in range(0, len(self.window) - 1):
    		sum += -2 * self.gate_opening * (np.subtract(self.window[t+1][:], self.O[t])).outer(self.S[:,t])
        return sum

    #-------------------------------------------
    def compute_activation(self, j, time):
        a = 0
        for i in range(0, nb_motor + nb_sensor):
                a += self.U[j][i] * self.window[time][i]
        for i in range(0, hidden_dim):
                a += self.W[j][i] * self.S[time - 1][i]
        return a

    #-------------------------------------------
    def compute_partial_s_j_t_U(self, j, A, B, time):
        # Compute values required to compute the partial derivative
        a = compute_activation(j, time)
        if(j == A):
            x_b = self.window[time][B]
        else:
            x_b = 0
        # Compute the partial derivative
        if(time == 0):
            return (1 - np.tanh(a)**2) * x_b
        else:
        	d_s_sum = 0
        	for k in range(0, hidden_dim):
        		d_s_sum += self.W[j][k] * compute_partial_s_j_t_U(k, A, B, time - 1)
			return (1 - np.tanh(a)**2) * (x_b + d_s_sum)
        
    #-------------------------------------------
    def compute_partial_E_U_A_B(self, A, B, time):
        sum = 0
        for i in range(0, nb_motor + nb_sensor):
            for j in range(0,hidden_dim):
                sum += -2 * self.gate_opening * (self.window[time + 1][i] - self.O[time][i]) * self.V[i][j] * compute_partial_s_j_t_U(j, A, B, time)
        return sum

    #-------------------------------------------
    def compute_partial_E_U(self):
        dE_dU = np.zeros( hidden_dim, nb_sensor + nb_motor)
        for A in range(0, hidden_dim):
            for B in range(0, nb_sensor + nb_motor):
                for t in range(0, len(self.window) - 1):
                    dE_dU[A][B] += compute_partial_E_U_A_B(A, B, t)
        return dE_dU

    #-------------------------------------------
    def compute_partial_s_j_t_W(self, j, A, B, time):
        # Compute values required to compute the partial derivative
        a = compute_activation(j, time)
        if(j == A):
            s_b = self.S[B, time]
        else:
            s_b = 0
        # Compute the partial derivative
        if(time == 0):
            return (1 - np.tanh(a)**2) * s_b
        else:
        	d_s_sum = 0
        	for k in range(0, hidden_dim):
        		d_s_sum += self.W[j][k] * compute_partial_s_j_t_W(k, A, B, time - 1)
			return (1 - np.tanh(a)**2) * (s_b + d_s_sum)
        
    #-------------------------------------------
    def compute_partial_E_W_A_B(self, A, B, time):
        sum = 0
        for i in range(0, nb_motor + nb_sensor):
            for j in range(0,hidden_dim):
                sum += -2 * self.gate_opening * (self.window[time + 1][i] - self.O[time][i]) * self.V[i][j] * compute_partial_s_j_t_W(j, A, B, time)
        return sum

    #-------------------------------------------
    def compute_partial_E_W(self):
        dE_dW = np.zeros( hidden_dim, nb_sensor + nb_motor)
        for A in range(0, hidden_dim):
            for B in range(0, nb_sensor + nb_motor):
                for t in range(0, len(self.window) - 1):
                    dE_dW[A][B] += compute_partial_E_W_A_B(A, B, t)
        return dE_dW

    #-------------------------------------------
    def update_weights(self):
   		self.W = np.add(self.W, self.learning_rate * compute_partial_E_W())
   		self.V = np.add(self.V, self.learning_rate * compute_partial_E_V())
   		self.U = np.add(self.U, self.learning_rate * compute_partial_E_U())










#-------------------------------------------
class ExpertMixture:
    def __init__(self, data, epsilon_g, nu_g, scaling, RNN_number=5):
        self.epsilon_g = epsilon_g
        self.nu_g = nu_g
        self.scaling = scaling
        self.data = data
        self.error = np.zeros((RNN_number, len(data[0])))
        self.s_t = collections.deque(maxlen=2)
        self.g_t = np.zeros(RNN_number)
        self.RNN_number = RNN_number
        # Create a dictionnary of RNN_number RNN_modules
        self.RNNs = {}
        for i in range(0, RNN_number):
        	self.RNNs["RNN_{0}".format(i)] = RNN_module(num=i, nb_sensor=6, nb_motor=2, scaling=scaling)

    #-------------------------------------------
    # Target et output contiennent les valeurs obtenues du robot/simulation et celles de sortie du RNN respectivement
    # Rangees dans des dictionnaires (sous la forme dict[k-ieme RNN][j-ieme composante de l'etat])
    def norm_2_i(self, index):
        norm = 0
        for i in range(0, len(self.error)):
            norm += (self.error[index][i])**2
        return norm
    #-------------------------------------------
    def compute_gt_i(self, index):
        sum = 0
        for i in range(0, len(self.s_t[1])):
            sum += math.exp(self.s_t[1][i])
        return (math.exp(self.s_t[1][index]) / sum)

    #-------------------------------------------
    def compute_post_proba(self, index):
        sum = 0
        for i in range(0, len(self.s_t[1])):
            sum += self.g_t[i] * math.exp((-1/(2*self.scaling))*self.norm_2_i(i))
        return (self.g_t[index] * math.exp((-1/(2*self.scaling))*self.norm_2_i(index)) / sum)

    #-------------------------------------------
    def compute_partial_L_s_k(self, index):
        return (self.compute_post_proba(index) - self.g_t[index])

    #-------------------------------------------
    def compute_delta_s_k_i(self, index):
        return (self.epsilon_g * self.compute_partial_L_s_k(index) - self.nu_g * (self.s_t[1][index] - self.s_t[0][index]))

    #-------------------------------------------
    def routine(self):
    	d={}
    	for number in range (1, self.RNN_number + 1):
        	d["error_{0}".format(number)] = rospy.Publisher('/MRE/error_'+str(number), Float32 , queue_size=10)

    	tmp = [0 for i in range(0,self.RNN_number)]
    	self.s_t.append(tmp)
    	for t in range(0, len(self.data) - 1):
	    	for i in range(0, self.RNN_number):
	    		o = self.RNNs["RNN_{0}".format(i)].predict(self.data[t])
    			self.error[i] = np.subtract(np.array(self.data[t]),o)
    			d["error_{0}".format(i + 1)].publish(self.norm_2_i(i))

	    		tmp[i] = self.RNNs["RNN_{0}".format(i)].get_gate_state()

	    	self.s_t.append(tmp)

	    	for i in range(0, self.RNN_number):
	    		self.g_t[i] = self.compute_gt_i(i)

	    	for i in range(0, self.RNN_number):
	    		tmp[i] += self.compute_delta_s_k_i(i)
	    		self.RNNs["RNN_{0}".format(i)].set_gate_state(self.s_t[1][i])
	    		self.s_t[0][i] = self.s_t[1][i]
	    	self.s_t.append(tmp)
	    	print t

    	





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
		return data[0:(time + 2)]			# Return everything we can according to current time
	elif(time > 0):
		return data[(time - window_size +2):time + 2] 	# Return a window vector of [t-2, t-1, t, t+1] according to current time
	print "Error in data_to_window: check time input"

#-------------------------------------------
def online_learning():
    rospy.init_node('online_learning', anonymous=True)

    # The node publishes the gating values so that rqt_plot can plot those :
    d={}
    # for number in range (1, 6):
    #     d["pub_gate_{0}".format(number)] = rospy.Publisher('/MRE/gate_'+str(number), Float32 , queue_size=10)

    # The node receives sensory and motor information from simu_fastsim:
    rospy.Subscriber("/simu_fastsim/lasers", Float32MultiArray, callback_lasers)
    rospy.Subscriber('/simu_fastsim/speed_left', Float32, callback_speed_left)
    rospy.Subscriber('/simu_fastsim/speed_right', Float32, callback_speed_right)

    # Targetted operating frequency of the node:
    r = rospy.Rate(rate) # 10hz

    # Import the data
    rospack = rospkg.RosPack()

    # Get the file path for rospy_tutorials
    path = rospack.get_path('RNN_python')

    # Read left motor data from csv file
    m_left = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_speed_left.csv', m_left, list_flag=False)

    # Read right motor data from csv file
    m_right = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_speed_right.csv', m_right, list_flag=False)

    # Read lasers data from csv file
    lasers = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_lasers.csv', lasers, list_flag=True)

    # Align the data
    training_data = align_data(m_left, m_right, lasers)

    # Start time and timing related things
    startT = rospy.get_time()
    rospy.loginfo("Start time: " + str(startT))

    mre = ExpertMixture(training_data, 0.007, 0.02, 100)
    mre.routine()
	    

    while (not rospy.is_shutdown()):
        # d["pub_gate_1"].publish(1)
        pass

    

#-------------------------------------------
if __name__ == '__main__':
    try:
        online_learning()
    except rospy.ROSInterruptException: pass