import math
from . import dp_functions as dp


class DifferentialPrivacy(object):
    data = []
    valid_data = []
    anon_data = []
    n = 0
    probability = 1/3
    epsilon = math.nan
    delta_f = 1.0
    delta_v = 1.0
    delta_method = sum

    def __init__(self, data: list = [], p: float = None, epsilon: float = None):
        self.data = data

        if not epsilon is None:
            self.epsilon = epsilon
            self.update(['data', 'epsilon'])
            self.probability = dp.probability(
                self.n, self.epsilon, self.delta_f, self.delta_v)
            return

        if not p is None:
            self.probability = p

        self.update(['data'])

    def update(self, props: list = []):
        if 'data' in props:
            self.valid_data = [
                value for value in self.data if not math.isnan(value)]
            self.n = len(self.valid_data)

        if len(self.valid_data) <= 1/self.probability:
            return

        if any(prop in ['data', 'delta_method'] for prop in props):
            self.delta_f = dp.delta_f(self.valid_data, self.delta_method)
            self.delta_v = dp.delta_v(self.valid_data, self.delta_method)

        if not 'epsilon' in props:
            self.epsilon = dp.epsilon(
                self.n, self.probability, self.delta_f, self.delta_v)

    @property
    def scale(self): return dp.scale(self.epsilon, self.delta_f)

    @property
    def privacy(self): return dp.privacy(self.anon_data, self.epsilon)

    @property
    def utility(self): return dp.utility(self.data, self.anon_data)

    def apply(self, callback: callable):
        return [callback(value) for value in self.data]

    def laplace(self):
        b = self.scale
        self.anon_data = self.apply(lambda value: dp.laplace(value, b))
        return self.anon_data
