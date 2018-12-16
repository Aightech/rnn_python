import rospy
from std_msgs.msg import Int16,Float32,Bool,Float32MultiArray,Int16MultiArray
import rospkg 
import csv

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

class RNNNumpy:
    
    def __init__(self,num, nb_sensor, nb_motor, hidden_dim=4, bptt_truncate=4):
        # Assign instance variables
        self.num = num
        self.gate_state = 0
        self.gate_opening = 0
        self.nb_sensor = nb_sensor
        self.nb_motor = nb_motor
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./(nb_sensor+nb_motor)), np.sqrt(1./(nb_sensor + nb_motor)), (hidden_dim, nb_sensor + nb_motor))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (nb_sensor + nb_motor, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def set_gate_state(self,s):
        self.gate_state = ds

    def get_gate_state(self):
        return self.gate_state

    def set_gate_opening(self,g):
        self.gate_opening = g

    def get_gate_opening(self):
        return self.gate_opening
    
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.nb_sensor + self.nb_motor))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U.dot(x[:,t]) + self.W.dot(s[t-1]))
            o[t] = self.V.dot(s[t])
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return o[-1]

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

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
def read_csv(path_to_file, dict_to_fill):
    with open(path_to_file, 'rb') as csvfile:
        next(csvfile)
        r = csv.reader(csvfile, delimiter=',')
        for row in r:
            a = (((row[-1].replace("[", "")).replace(" ", "")).replace("]", "")).split(',')
            b = [float(i) for i in a]
            c = [100.0 if x == -1 else x for x in b]
            dict_to_fill[float(row[0])] = c
            
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
    read_csv(path + '/data/_slash_simu_fastsim_slash_speed_left.csv', m_left)

    # Read right motor data from csv file
    m_right = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_speed_right.csv', m_right)

    # Read lasers data from csv file
    lasers = {}
    read_csv(path + '/data/_slash_simu_fastsim_slash_lasers.csv', lasers)
    print lasers
    # Start time and timing related things
    startT = rospy.get_time()
    rospy.loginfo("Start time: " + str(startT))

    while (not rospy.is_shutdown()):
        # d["pub_gate_1"].publish(1)
        pass

    # rnn = RNNNumpy(nb_sensor,nb_motor,4)
    # print(rnn.forward_propagation(Xpredict))
    # print(rnn.U)

#-------------------------------------------
if __name__ == '__main__':
    try:
        online_learning()
    except rospy.ROSInterruptException: pass