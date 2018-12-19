## RNN Network
### Structure
![alt text](img/rnn.png "RNN Network")

### Dimensions
<a href="https://www.codecogs.com/eqnedit.php?latex=X\in&space;\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X\in&space;\mathbb{R}^{n}" title="X\in \mathbb{R}^{n}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Y\in&space;\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y\in&space;\mathbb{R}^{n}" title="Y\in \mathbb{R}^{n}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=S\in&space;\mathbb{R}^{d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S\in&space;\mathbb{R}^{d}" title="S\in \mathbb{R}^{d}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Z\in&space;\mathbb{R}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z\in&space;\mathbb{R}^{n}" title="Z\in \mathbb{R}^{n}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=U\in&space;\mathbb{R}^{d&space;x&space;n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U\in&space;\mathbb{R}^{d&space;x&space;n}" title="U\in \mathbb{R}^{d x n}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=V\in&space;\mathbb{R}^{n&space;x&space;d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V\in&space;\mathbb{R}^{n&space;x&space;d}" title="V\in \mathbb{R}^{n x d}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=W\in&space;\mathbb{R}^{d&space;x&space;d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W\in&space;\mathbb{R}^{d&space;x&space;d}" title="W\in \mathbb{R}^{d x d}" /></a>


### Relations
<a href="https://www.codecogs.com/eqnedit.php?latex=Z^t=U.X^t&space;&plus;&space;W.S^{t-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z^t=U.X^t&space;&plus;&space;W.S^{t-1}" title="Z^t=U.X^t + W.S^{t-1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=S^t=f(Z^t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S^t=f(Z^t)" title="S^t=f(Z^t)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=f&space;=&space;tanh" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;=&space;tanh" title="f = tanh" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Y^t=V.S^t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y^t=V.S^t" title="Y^t=V.S^t" /></a>


### Gradient

#### U
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E^T}{\partial&space;U}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;u_{ij}}\right)_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E^T}{\partial&space;U}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;u_{ij}}\right)_{ij}" title="\frac{\partial E^T}{\partial U} = \left ( \sum_{t=1}^{T} \frac{\partial E^t}{\partial u_{ij}}\right)_{ij}" /></a>


<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E^t}{\partial&space;u_{ij}}&space;=&space;\sum_{k=1}^{n}\sum_{l=1}^{d}-2g(\hat{y}^t_k-y^t_k)&space;.&space;v_{kl}&space;.&space;\frac{\partial&space;s^t_l}{\partial&space;u_{ij}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E^t}{\partial&space;u_{ij}}&space;=&space;\sum_{k=1}^{n}\sum_{l=1}^{d}-2g(\hat{y}^t_k-y^t_k)&space;.&space;v_{kl}&space;.&space;\frac{\partial&space;s^t_l}{\partial&space;u_{ij}}" title="\frac{\partial E^t}{\partial u_{ij}} = \sum_{k=1}^{n}\sum_{l=1}^{d}-2g(\hat{y}^t_k-y^t_k) . v_{kl} . \frac{\partial s^t_l}{\partial u_{ij}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;s_l^0}{\partial&space;u_{ij}}&space;=&space;0&space;\;\;\;\;&space;et&space;\;\;\;\;&space;\forall&space;t&space;\in&space;[0,T],&space;\;\frac{\partial&space;s_l^t}{\partial&space;u_{ij}}&space;=&space;f'(z^t_l).\left(x^t_j.\delta_{li}&space;&plus;&space;\sum_{m=1}^{d}w_{lm}\frac{\partial&space;s^t_m}{\partial&space;u_{ij}}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;s_l^0}{\partial&space;u_{ij}}&space;=&space;0&space;\;\;\;\;&space;et&space;\;\;\;\;&space;\forall&space;t&space;\in&space;[0,T],&space;\;\frac{\partial&space;s_l^t}{\partial&space;u_{ij}}&space;=&space;f'(z^t_l).\left(x^t_j.\delta_{li}&space;&plus;&space;\sum_{m=1}^{d}w_{lm}\frac{\partial&space;s^t_m}{\partial&space;u_{ij}}&space;\right)" title="\frac{\partial s_l^0}{\partial u_{ij}} = 0 \;\;\;\; et \;\;\;\; \forall t \in [0,T], \;\frac{\partial s_l^t}{\partial u_{ij}} = f'(z^t_l).\left(x^t_j.\delta_{li} + \sum_{m=1}^{d}w_{lm}\frac{\partial s^t_m}{\partial u_{ij}} \right)" /></a>

#### V
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E^T}{\partial&space;V}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;v_{ij}}\right)_{ij}&space;=&space;\left(&space;\sum_{t=1}^{T}\frac{\partial&space;E^t}{\partial&space;y_i}&space;.&space;\frac{\partial&space;y_i}{\partial&space;v_{ij}}&space;\right)_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E^T}{\partial&space;V}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;v_{ij}}\right)_{ij}&space;=&space;\left(&space;\sum_{t=1}^{T}\frac{\partial&space;E^t}{\partial&space;y_i}&space;.&space;\frac{\partial&space;y_i}{\partial&space;v_{ij}}&space;\right)_{ij}" title="\frac{\partial E^T}{\partial V} = \left ( \sum_{t=1}^{T} \frac{\partial E^t}{\partial v_{ij}}\right)_{ij} = \left( \sum_{t=1}^{T}\frac{\partial E^t}{\partial y_i} . \frac{\partial y_i}{\partial v_{ij}} \right)_{ij}" /></a>
