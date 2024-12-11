import random

class Memory:
    def __init__(self, max_size, min_size):
        self.samples = []  
        self.max_size = max_size  
        self.min_size = min_size  

    def add_sample(self, sample):
        self.samples.append(sample)
        if self.current_size() > self.max_size:
            self.samples.pop(0)

    def get_samples(self, n):
        if self.current_size() < self.min_size:
            return []

        return random.sample(self.samples, min(n, self.current_size()))

    def current_size(self):
        return len(self.samples)
