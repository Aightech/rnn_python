import rospy
from std_msgs.msg import Int16,Float32,Bool,Float32MultiArray,Int16MultiArray
import rospkg 
import csv
import collections

import numpy as np

# Read from topics values
lasers = list()
speed_left = 0
speed_right = 0

# Rate for ROS in Hz
rate = 10


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

class RNN:
    
    def __init__(self,num, nb_sensor, nb_motor, max_window_size, hidden_dim=4, bptt_truncate=4):
        # Assign instance variables
        self.num = num
        self.gate_state = 0
        self.gate_opening = 0
        self.nb_sensor = nb_sensor
        self.nb_motor = nb_motor
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Window variables
        self.max_window_size = max_window_size
        self.window = collections.deque(maxlen = max_window_size)
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./(nb_sensor+nb_motor)), np.sqrt(1./(nb_sensor + nb_motor)), (hidden_dim, nb_sensor + nb_motor))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (nb_sensor + nb_motor, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.S = np.zeros( hidden_dim, max_window_size + 1)


    #-------------------------------------------
    def set_gate_state(self,s):
        self.gate_state = s

    #-------------------------------------------
    def get_gate_state(self):
        return self.gate_state

    #-------------------------------------------
    def set_gate_opening(self,g):
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
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        self.S[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.nb_sensor + self.nb_motor))
        # For each time step...
        for t in np.arange(T):
            self.S[:,t] = np.tanh(self.U.dot(x[:,t]) + self.W.dot(self.S[:,t-1]))
            o[t] = self.V.dot(self.S[:,t])
        return o

    #-------------------------------------------
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o = self.forward_propagation(x)
        return o[-1]

    #-------------------------------------------
    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    #-------------------------------------------
    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    #-------------------------------------------
    def compute_partial_E_V(target, output, time):
        return -2 * self.gate_opening * (np.subtract(target[t], output)).dot(self.S[t]) # Attention, a verifier au niveau de la multiplication

    #-------------------------------------------
    def compute_activation_i(x, i, time):
        a = 0
        for j in range(0, nb_motor + nb_sensor):
                a += self.U[i][j] * x[j]
        for j in range(0, hidden_dim):
                a += self.W[i][j] * self.S[time - 1][j]
        return a

    #-------------------------------------------
    def compute_partial_s_j_t_U(target, output, x, i, A, B, time):
        # Compute values required to compute the partial derivative
        a = compute_activation(x, i, time)
        if(i == A):
            x_b = x[B]
        else:
            x_b = 0
        # Compute the partial derivative
        if(time == 0):
            return (1 - np.tanh(a)**2) * x_b
        else:
            return (1 - np.tanh(a)**2) * (x_b + compute_partial_s_j_t_U(target, output, x, i, A, B, time - 1))
        
    #-------------------------------------------
    def compute_partial_E_U_A_B(target, output, x, A, B, time):
        sum = 0
        for i in range(0, nb_motor + nb_sensor):
            for j in range(0,hidden_dim):
                sum += 2 * self.gate_opening * (target[time][i] - output[i]) * self.V[i][j] * compute_partial_s_j_t_U(target, output, x, j, A, B, time)
        return sum
    #-------------------------------------------
    def compute_partial_E_U(target, output, x, time):
        dE_dU = np.zeros( hidden_dim, nb_sensor + nb_motor)
        for A in range(0, hidden_dim):
            for B in range(0, nb_sensor + nb_motor):
                for t in range(0, len(self.window)):
                    dE_dU[A][B] += compute_partial_E_U_A_B(target, output, x, A, B, t)
        return dE_dU

class ExpertMixture:
    def __init__(self, epsilon_g, nu_g, scaling):
        self.epsilon_g = epsilon_g
        self.nu_g = nu_g
        self.scaling = scaling

    #-------------------------------------------
    # Target et output contienne les valeurs obtenues du robot/simulation et celles de sortie du RNN respectivement
    # Rangees dans des dictionnaires (sous la forme dict[k-ieme RNN][j-ieme composante de l'etat])
    def norm_2_i(target, output, index):
        if(len(target) != len(output)):
            print "Erreur norm_2_i: taille des tableaux target et output non identique"
        norm = 0
        for i in range(0, len(target)):
            norm += (target[index][i] - output[index][i])**2
        return norm
    #-------------------------------------------
    # We need a fonction to get s_t from each RNN
    def compute_gt_i(s_t, index):
        sum = 0
        for i in range(0, len(s_t)):
            sum += math.exp(s_t[i])
        return (math.exp(s_t[index]) / sum)

    #-------------------------------------------
    def compute_post_proba(s_t, output, target, index):
        sum = 0
        for i in range(0, len(s_t)):
            sum+= compute_gt_i(s_t, i) * math.exp((-1/(2*self.scaling))*norm_2_i(target, output, i))
        return (compute_gt_i(s_t, index) * math.exp((-1/(2*self.scaling))*norm_2_i(target, output, index)) / sum)

    #-------------------------------------------
    def compute_partial_L_s_k(s_k, output, target, index):
        return (compute_post_proba(s_k, output, target, index) - compute_gt_i(s_k, index))

    #-------------------------------------------
    def compute_delta_s_k_i(s_k, sk_1, output, target, index):
        return (self.epsilon_g * compute_partial_L_s_k(s_k, output, target, index) - self.nu_g * (s_k[i] - s_k_1[i]))

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
def align_data(m_left, m_right, lasers):
    k = 0
    data = []
    for i in range(0, len(lasers)):
        if(m_left[k][0] < lasers[i][0] and k != len(m_left) - 1):
            k += 1
        data.append([m_left[k][1], m_right[k][1]] + [a for a in lasers[i][1]])
    return data

#-------------------------------------------
def online_learning():
    rospy.init_node('online_learning', anonymous=True)

    # The node publishes the gating values so that rqt_plot can plot those :
    d={}
    for number in range (1, 6):
        d["pub_gate_{0}".format(number)] = rospy.Publisher('/MRE/gate_'+str(number), Float32 , queue_size=10)

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

    while (not rospy.is_shutdown()):
        # d["pub_gate_1"].publish(1)
        pass

    # rnn = RNN(nb_sensor,nb_motor,4)
    # print(rnn.forward_propagation(Xpredict))
    # print(rnn.U)

#-------------------------------------------
if __name__ == '__main__':
    try:
        online_learning()
    except rospy.ROSInterruptException: pass