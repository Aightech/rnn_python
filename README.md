## RNN Network
### Structure
![alt text](img/rnn.png "RNN Network")

### Dimensions
<a href="https://www.codecogs.com/eqnedit.php?latex=$X\in&space;\mathbb{R}^{n}&space;$&space;\;,\;&space;$Y\in&space;\mathbb{R}^{n}&space;$&space;\;,\;&space;$S\in&space;\mathbb{R}^{d}&space;$&space;\;,\;&space;$Z\in&space;\mathbb{R}^{n}&space;$&space;\;,\;&space;$U\in&space;\mathbb{R}^{d&space;.&space;n}&space;$&space;\;,\;&space;$V\in&space;\mathbb{R}^{n&space;.&space;d}&space;$&space;\;,\;&space;$W\in&space;\mathbb{R}^{d&space;.&space;d}&space;$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$X\in&space;\mathbb{R}^{n}&space;$&space;\;,\;&space;$Y\in&space;\mathbb{R}^{n}&space;$&space;\;,\;&space;$S\in&space;\mathbb{R}^{d}&space;$&space;\;,\;&space;$Z\in&space;\mathbb{R}^{n}&space;$&space;\;,\;&space;$U\in&space;\mathbb{R}^{d&space;.&space;n}&space;$&space;\;,\;&space;$V\in&space;\mathbb{R}^{n&space;.&space;d}&space;$&space;\;,\;&space;$W\in&space;\mathbb{R}^{d&space;.&space;d}&space;$" title="$X\in \mathbb{R}^{n} $ \;,\; $Y\in \mathbb{R}^{n} $ \;,\; $S\in \mathbb{R}^{d} $ \;,\; $Z\in \mathbb{R}^{n} $ \;,\; $U\in \mathbb{R}^{d . n} $ \;,\; $V\in \mathbb{R}^{n . d} $ \;,\; $W\in \mathbb{R}^{d . d} $" /></a>


### Relations
<a href="https://www.codecogs.com/eqnedit.php?latex=$Y^t=V.S^t$&space;\;\;&space;et&space;\;\;&space;$S^t=f(Z^t)$&space;\;\;&space;avec&space;\;\&space;$Z^t=U.X^t&space;&plus;&space;W.S^{t-1}$&space;\;\;&space;et&space;\;\;&space;$f&space;=&space;tanh$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$Y^t=V.S^t$&space;\;\;&space;et&space;\;\;&space;$S^t=f(Z^t)$&space;\;\;&space;avec&space;\;\&space;$Z^t=U.X^t&space;&plus;&space;W.S^{t-1}$&space;\;\;&space;et&space;\;\;&space;$f&space;=&space;tanh$" title="$Y^t=V.S^t$ \;\; et \;\; $S^t=f(Z^t)$ \;\; avec \;\ $Z^t=U.X^t + W.S^{t-1}$ \;\; et \;\; $f = tanh$" /></a>


### Gradients
---
#### U
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E^T}{\partial&space;U}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;u_{ij}}\right)_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E^T}{\partial&space;U}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;u_{ij}}\right)_{ij}" title="\frac{\partial E^T}{\partial U} = \left ( \sum_{t=1}^{T} \frac{\partial E^t}{\partial u_{ij}}\right)_{ij}" /></a>


<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E^t}{\partial&space;u_{ij}}&space;=&space;\sum_{k=1}^{n}\sum_{l=1}^{d}-2g(\hat{y}^t_k-y^t_k)&space;.&space;v_{kl}&space;.&space;\frac{\partial&space;s^t_l}{\partial&space;u_{ij}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E^t}{\partial&space;u_{ij}}&space;=&space;\sum_{k=1}^{n}\sum_{l=1}^{d}-2g(\hat{y}^t_k-y^t_k)&space;.&space;v_{kl}&space;.&space;\frac{\partial&space;s^t_l}{\partial&space;u_{ij}}" title="\frac{\partial E^t}{\partial u_{ij}} = \sum_{k=1}^{n}\sum_{l=1}^{d}-2g(\hat{y}^t_k-y^t_k) . v_{kl} . \frac{\partial s^t_l}{\partial u_{ij}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;s_l^0}{\partial&space;u_{ij}}&space;=&space;0&space;\;\;\;\;&space;et&space;\;\;\;\;&space;\forall&space;t&space;\in&space;[0,T],&space;\;\frac{\partial&space;s_l^t}{\partial&space;u_{ij}}&space;=&space;f'(z^t_l).\left(x^t_j.\delta_{li}&space;&plus;&space;\sum_{m=1}^{d}w_{lm}\frac{\partial&space;s^t_m}{\partial&space;u_{ij}}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;s_l^0}{\partial&space;u_{ij}}&space;=&space;0&space;\;\;\;\;&space;et&space;\;\;\;\;&space;\forall&space;t&space;\in&space;[0,T],&space;\;\frac{\partial&space;s_l^t}{\partial&space;u_{ij}}&space;=&space;f'(z^t_l).\left(x^t_j.\delta_{li}&space;&plus;&space;\sum_{m=1}^{d}w_{lm}\frac{\partial&space;s^t_m}{\partial&space;u_{ij}}&space;\right)" title="\frac{\partial s_l^0}{\partial u_{ij}} = 0 \;\;\;\; et \;\;\;\; \forall t \in [0,T], \;\frac{\partial s_l^t}{\partial u_{ij}} = f'(z^t_l).\left(x^t_j.\delta_{li} + \sum_{m=1}^{d}w_{lm}\frac{\partial s^t_m}{\partial u_{ij}} \right)" /></a>
---
#### V
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E^T}{\partial&space;V}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;v_{ij}}\right)_{ij}&space;=&space;\left(&space;\sum_{t=1}^{T}\frac{\partial&space;E^t}{\partial&space;y_i}&space;.&space;\frac{\partial&space;y_i}{\partial&space;v_{ij}}&space;\right)_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E^T}{\partial&space;V}&space;=&space;\left&space;(&space;\sum_{t=1}^{T}&space;\frac{\partial&space;E^t}{\partial&space;v_{ij}}\right)_{ij}&space;=&space;\left(&space;\sum_{t=1}^{T}\frac{\partial&space;E^t}{\partial&space;y_i}&space;.&space;\frac{\partial&space;y_i}{\partial&space;v_{ij}}&space;\right)_{ij}" title="\frac{\partial E^T}{\partial V} = \left ( \sum_{t=1}^{T} \frac{\partial E^t}{\partial v_{ij}}\right)_{ij} = \left( \sum_{t=1}^{T}\frac{\partial E^t}{\partial y_i} . \frac{\partial y_i}{\partial v_{ij}} \right)_{ij}" /></a>


## Results
**Several Experts**
![alt text](img/gating_1.png "Gates: Shared")
**Only one expert**
![alt text](img/gating_2.png "Gates: Super Expert")

### Several layers
**Low level getes**
![alt text](img/gating_low_1.png "Gates low level")
**High level gates**
![alt text](img/gating_high_1.png "Gates high level")
