class DCGANNormalize(object):
    
    def __call__(self, sample):
        return (sample - 0.5) * 2