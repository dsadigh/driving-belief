import numpy as np
import utils
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
from trajectory import Trajectory
import feature

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        self.data0 = {'x0': x0}
        self.T = T
        self.dyn = dyn
        self.traj = Trajectory(T, dyn)
        self.traj.x0.set_value(x0)
        self.x_hist = [x0]*T
        self.past = Trajectory(T, dyn)
        self.past.x0.set_value(x0)
        self.linear = Trajectory(T, dyn)
        self.linear.x0.set_value(x0)
        self.color = color
        self.default_u = np.zeros(self.dyn.nu)
    def reset(self):
        self.x_hist = [self.data0['x0']]*self.T
        self.traj.x0.set_value(self.data0['x0'])
        self.past.x0.set_value(self.data0['x0'])
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):
            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.past.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def past_tick(self):
        self.x_hist = self.x_hist[1:]+[self.x]
        self.past.tick()
        self.past.x0.set_value(self.x_hist[0])
        self.past.u[self.T-1].set_value(self.u)
    def move(self):
        self.past_tick()
        self.traj.tick()
        self.linear.x0.set_value(self.traj.x0.get_value())
    @property
    def x(self):
        return self.traj.x0.get_value()
    @property
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter
    def u(self, value):
        self.traj.u[0].set_value(value)
    def control(self, steer, gas):
        pass

class UserControlledCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
    def control(self, steer, gas):
        self.u = [steer, gas]

class SimpleOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
    @property
    def reward(self):
        return self._reward
    @reward.setter
    def reward(self, reward):
        self._reward = reward
        self.optimizer = None
    def control(self, steer, gas):
        if self.optimizer is None:
            r = self.traj.total(self.reward)
            self.optimizer = utils.Maximizer(r, self.traj.u)
        self.optimizer.maximize()

class NestedOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
    @property
    def human(self):
        return self._human
    @human.setter
    def human(self, value):
        self._human = value
        self.traj_h = Trajectory(self.T, self.human.dyn)
    def move(self):
        Car.move(self)
        self.traj_h.tick()
    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, vals):
        self._rewards = vals
        self.optimizer = None
    def control(self, steer, gas):
        import ipdb; ipdb.set_trace()
        if self.optimizer is None:
            reward_h, reward_r = self.rewards
            reward_h = self.traj_h.total(reward_h)
            reward_r = self.traj.total(reward_r)
            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
        self.traj_h.x0.set_value(self.human.x)
        self.optimizer.maximize(bounds = self.bounds)

class BeliefOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
        self.dumb = False
    @property
    def human(self):
        return self._human
    @human.setter
    def human(self, value):
        self._human = value
        self.traj_hs = []
        self.log_ps = []
        self.rewards = []
        self.optimizer = None
    def add_model(self, reward, log_p=0.):
        self.traj_hs.append(Trajectory(self.T, self.human.dyn))
        weight = utils.scalar()
        weight.set_value(log_p)
        self.log_ps.append(weight)
        self.rewards.append(reward)
        self.data0['log_ps'] = [log_p.get_value() for log_p in self.log_ps]
        self.optimizer = None
    @property
    def objective(self):
        return self._objective
    @objective.setter
    def objective(self, value):
        self._objective = value
        self.optimizer = None
    def reset(self):
        Car.reset(self)
        for log_p, val in zip(self.log_ps, self.data0['log_ps']):
            log_p.set_value(val)
        if hasattr(self, 'normalize'):
            self.normalize()
        self.t = 0
        if self.dumb:
            self.useq = self.objective
    def move(self):
        Car.move(self)
        self.t += 1
    def entropy(self, traj_h):
        new_log_ps = [traj_h.log_p(reward('traj'))+log_p for log_p, reward in zip(self.log_ps, self.rewards)]
        mean_log_p = sum(new_log_ps)/len(new_log_ps)
        new_log_ps = [x-mean_log_p for x in new_log_ps]
        s = tt.log(sum(tt.exp(x) for x in new_log_ps))
        new_log_ps = [x-s for x in new_log_ps]
        return sum(x*tt.exp(x) for x in new_log_ps)
    def control(self, steer, gas):
        if self.optimizer is None:
            u = sum(log_p for log_p in self.log_ps)/len(self.log_ps)
            self.prenormalize = th.function([], None, updates=[(log_p, log_p-u) for log_p in self.log_ps])
            s = tt.log(sum(tt.exp(log_p) for log_p in self.log_ps))
            self.normalize = th.function([], None, updates=[(log_p, log_p-s) for log_p in self.log_ps])
            self.update_belief = th.function([], None, updates=[(log_p, log_p + self.human.past.log_p(reward('past'))) for reward, log_p in zip(self.rewards, self.log_ps)])
            self.normalize()
            self.t = 0
            if self.dumb:
                self.useq = self.objective
                self.optimizer = True
            else:
                if hasattr(self.objective, '__call__'):
                    obj_h = sum([traj_h.total(reward('traj')) for traj_h, reward in zip(self.traj_hs, self.rewards)])
                    var_h = sum([traj_h.u for traj_h in self.traj_hs], [])
                    obj_r = sum(tt.exp(log_p)*self.objective(traj_h) for traj_h, log_p in zip(self.traj_hs, self.log_ps))
                    self.optimizer = utils.NestedMaximizer(obj_h, var_h, obj_r, self.traj.u)
                else:
                    obj_r = self.objective
                    self.optimizer = utils.Maximizer(self.objective, self.traj.u)
        if self.t == self.T:
            self.update_belief()
            self.t = 0
        if self.dumb:
            self.u = self.useq[0]
            self.useq = self.useq[1:]
        if self.t == 0:
            self.prenormalize()
            self.normalize()
            for traj_h in self.traj_hs:
                traj_h.x0.set_value(self.human.x)
            if not self.dumb:
                self.optimizer.maximize(bounds = self.bounds)
        for log_p in self.log_ps:
            print '%.2f'%np.exp(log_p.get_value()),
        print
        #for traj in self.traj_hs:
        #    traj.x0.set_value(self.human.x)
        #self.optimizer.maximize(bounds = self.bounds)
