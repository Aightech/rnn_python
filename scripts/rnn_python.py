#!/usr/bin/env python

import collections
import math
import numpy as np

class RNN_module:

    def __init__(self, num, nb_inputs, scaling, learning_rate=0.002, momentum=0.9, hidden_dim=4, bptt_truncate=4):
        # Assign instance variables
        self.num = num
        self.gate_state = 0
        self.gate_opening = 0
        self.nb_inputs = nb_inputs
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate
        # Window variables
        self.window = collections.deque(maxlen = bptt_truncate + 1)
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./nb_inputs)*scaling, np.sqrt(1./nb_inputs)*scaling, (hidden_dim, nb_inputs))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim)*scaling, np.sqrt(1./hidden_dim)*scaling, (nb_inputs, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim)*scaling, np.sqrt(1./hidden_dim)*scaling, (hidden_dim, hidden_dim))
        self.S = np.zeros( (hidden_dim, bptt_truncate + 1) )
        self.O = np.zeros( (nb_inputs, bptt_truncate + 1))

    #-------------------------------------------
    # x: input of the RNN (sensori-motor state at time t)
    # time: current time step
    def predict(self, x):
        self.append_value_window(x)
        self.O = self.forward_propagation()
        return self.get_output()

    #-------------------------------------------
    # Updates attributes of RNN class
    def update_gate(self, gate_opening, gate_state):
        self.set_gate_opening(gate_opening)
        self.set_gate_state(gate_state)

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
            return np.zeros(self.nb_inputs)

    #-------------------------------------------
    def forward_propagation(self):
        # The total number of time steps (we omit t + 1)
        T = len(self.window) - 1
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        self.S[:,-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((self.nb_inputs, T))
        # For each time step...
        for t in range(0, T):
            self.S[:,t] = np.tanh(self.U.dot(self.window[t][:]) + self.W.dot(self.S[:,t-1]))
            output[:,t] = self.V.dot(self.S[:,t])
        # Normalization of the output
        for i in range(0, len(output)):
            sum = 0
            for j in output[i]:
                sum += j**2
            output[i] = output[i]/math.sqrt(sum)
        return output   # Return output on the current time window

    #-------------------------------------------
    def compute_partial_E_V(self):
        sum = 0
        for t in range(0, len(self.window) - 1):
            sum += -2 * self.gate_opening * np.outer(np.subtract(self.window[t+1][:], self.O[:,t]), self.S[:,t])
        return sum

    #-------------------------------------------
    def compute_activation(self, j, time):
        a = 0
        for i in range(0, self.nb_inputs):
                a += self.U[j][i] * self.window[time][i]
        for i in range(0, self.hidden_dim):
                # print self.S
                a += self.W[j][i] * self.S[time - 1][i]
        return a

    #-------------------------------------------
    def compute_partial_s_j_t_U(self, j, A, B, time):
        # Compute values required to compute the partial derivative
        a = self.compute_activation(j, time)
        if(j == A):
            x_b = self.window[time][B]
        else:
            x_b = 0
        # Compute the partial derivative
        if(time == 0):
            return (1 - np.tanh(a)**2) * x_b
        else:
            d_s_sum = 0
            for k in range(0, self.hidden_dim):
                d_s_sum += self.W[j][k] * self.compute_partial_s_j_t_U(k, A, B, time - 1)
            return (1 - np.tanh(a)**2) * (x_b + d_s_sum)
        
    #-------------------------------------------
    def compute_partial_E_U_A_B(self, A, B, time):
        sum = 0
        for i in range(0, self.nb_inputs):
            for j in range(0, self.hidden_dim):
                sum += -2 * self.gate_opening * (self.window[time + 1][i] - self.O[i][time]) * self.V[i][j] * self.compute_partial_s_j_t_U(j, A, B, time)
        return sum

    #-------------------------------------------
    def compute_partial_E_U(self):
        dE_dU = np.zeros( (self.hidden_dim, self.nb_inputs) )
        for A in range(0, self.hidden_dim):
            for B in range(0, self.nb_inputs):
                for t in range(0, len(self.window) - 1):
                    dE_dU[A][B] += self.compute_partial_E_U_A_B(A, B, t)
        return dE_dU

    #-------------------------------------------
    def compute_partial_s_j_t_W(self, j, A, B, time):
        # Compute values required to compute the partial derivative
        a = self.compute_activation(j, time)
        if(j == A):
            s_b = self.S[B, time]
        else:
            s_b = 0
        # Compute the partial derivative
        if(time == 0):
            return (1 - np.tanh(a)**2) * s_b
        else:
            d_s_sum = 0
            for k in range(0, self.hidden_dim):
                d_s_sum += self.W[j][k] * self.compute_partial_s_j_t_W(k, A, B, time - 1)
            return (1 - np.tanh(a)**2) * (s_b + d_s_sum)
        
    #-------------------------------------------
    def compute_partial_E_W_A_B(self, A, B, time):
        sum = 0
        for i in range(0, self.nb_inputs):
            for j in range(0, self.hidden_dim):
                sum += -2 * self.gate_opening * (self.window[time + 1][i] - self.O[i][time]) * self.V[i][j] * self.compute_partial_s_j_t_W(j, A, B, time)
        return sum

    #-------------------------------------------
    def compute_partial_E_W(self):
        dE_dW = np.zeros( (self.hidden_dim, self.hidden_dim) )
        for A in range(0, self.hidden_dim):
            for B in range(0, self.hidden_dim):
                for t in range(0, len(self.window) - 1):
                    dE_dW[A][B] += self.compute_partial_E_W_A_B(A, B, t)
        return dE_dW

    #-------------------------------------------
    def update_weights(self):
        self.W = np.add(self.W, self.learning_rate * self.compute_partial_E_W())
        self.V = np.add(self.V, self.learning_rate * self.compute_partial_E_V())
        self.U = np.add(self.U, self.learning_rate * self.compute_partial_E_U())