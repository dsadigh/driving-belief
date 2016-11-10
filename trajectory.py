import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
import theano.tensor.nlinalg as tn
import utils
import numpy as np
import feature
import lane

class Trajectory(object):
    def __init__(self, T, dyn):
        self.dyn = dyn
        self.T = T
        self.x0 = utils.vector(dyn.nx)
        self.u = [utils.vector(dyn.nu) for t in range(self.T)]
        self.x = []
        z = self.x0
        for t in range(T):
            z = dyn(z, self.u[t])
            self.x.append(z)
        self.next_x = th.function([], self.x[0])
    def swap(self, traj):
        x0 = traj.x0.get_value()
        traj.x0.set_value(self.x0.get_value())
        self.x0.set_value(x0)
        for t in range(T):
            u = traj.u[t].get_value()
            traj.u[t].set_value(self.u[t].get_value())
            self.u[t].set_value(u)
    def tick(self):
        self.x0.set_value(self.next_x())
        for t in range(self.T-1):
            self.u[t].set_value(self.u[t+1].get_value())
        self.u[self.T-1].set_value(np.zeros(self.dyn.nu))
    def gaussian(self, height=.07, width=.03):
        @feature.feature
        def f(t, x, u):
            d = (self.x[t][0]-x[0], self.x[t][1]-x[1])
            theta = self.x[t][2]
            dh = tt.cos(theta)*d[0]+tt.sin(theta)*d[1]
            dw = -tt.sin(theta)*d[0]+tt.cos(theta)*d[1]
            return tt.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))
        return f
    def total(self, reward):
        r = [reward(t, self.x[t], self.u[t]) for t in range(self.T)]
        return sum(r)
    def log_p(self, reward):
        r = self.total(reward)
        g = utils.grad(r, self.u)
        H = utils.jacobian(g, self.u)
        return 0.5*tt.dot(g, tt.dot(tn.matrix_inverse(H), g))+0.5*tt.log(abs(tn.det(-H)))
        #return 0.5*tt.dot(g, ts.solve(H, g))+0.5*tt.log(abs(tn.det(-H)))
