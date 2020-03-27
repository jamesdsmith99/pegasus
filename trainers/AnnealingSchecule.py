class ConstantSchedule:

    def __init__(self, c=1):
        self.c = c

    def __call__(self, epoch):
        return self.c

class MonotonicSchedule:

    def  __init__(self, thresh, grad):
        self.thresh = thresh
        self.grad = grad

    def __call__(self, epoch):
        if epoch < self.thresh:
            return 0
        return min(1, self.grad*(epoch-self.thresh))

class CyclicSchedule:
    
    def __init__(self, slope_epochs, flat_epochs):
        self.slope_epochs = slope_epochs
        self.flat_epochs = flat_epochs

        self.cycle_length = slope_epochs + flat_epochs

        self.grad = 1 / slope_epochs

    def __call__(self, epoch):
        epoch %= self.cycle_length

        if epoch < self.slope_epochs:
            return epoch * self.grad
        return 1