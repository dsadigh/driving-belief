import theano as th
import theano.tensor as tt

class Feature(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args):
        return self.f(*args)
    def __add__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: self(*args)+r(*args))
        else:
            return Feature(lambda *args: self(*args)+r)
    def __radd__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: r(*args)+self(*args))
        else:
            return Feature(lambda *args: r+self(*args))
    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)
    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))
    def __pos__(self, r):
        return self
    def __neg__(self):
        return Feature(lambda *args: -self(*args))
    def __sub__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: self(*args)-r(*args))
        else:
            return Feature(lambda *args: self(*args)-r)
    def __rsub__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: r(*args)-self(*args))
        else:
            return Feature(lambda *args: r-self(*args))

def feature(f):
    return Feature(f)

def speed(s=1.):
    @feature
    def f(t, x, u):
        return -(x[3]-s)*(x[3]-s)
    return f

def control(bounds, width=0.05):
    @feature
    def f(t, x, u):
        ret = 0.
        for i, (a, b) in enumerate(bounds):
            if a is not None:
                ret += -tt.exp((a-u[i])/width)
            if b is not None:
                ret += -tt.exp((u[i]-b)/width)
        return ret
    return f

if __name__ == '__main__':
    pass
