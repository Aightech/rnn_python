import numpy as np

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

   #RNNNumpy.forward_propagation = forward_propagatino

rnn = RNNNumpy(nb_sensor,nb_motor,4)
print(rnn.forward_propagation(Xpredict))
print(rnn.U)

