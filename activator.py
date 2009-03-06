class Activator:
    "Activator function class"

    def __init__(self):
        self.error_hidden_unit = None
        self.error_output_unit = None
        pass

    def set_error_hidden_Unit(self, error):
        self.error_hidden_unit = error

    def set_error_output_Unit(self, error):
        self.error_output_unit = error

    def compute_activation(self, inputs, weights):
        return None

    def compute_derivative(self, x):
        return None 

    def compute_errors(self, is_output_unit, next_unit_errors, desired_output, outputs, next_unit_weights):
        if is_output_unit == True:
            error = self.error_output_unit
        else:
            error = self.error_hidden_unit

        return error.compute_errors(next_unit_errors, desired_output, outputs, next_unit_weights, self.compute_derivative)
 
class ActivatorLinear(Activator):
    "Linear activation function class"

    def __init__(self):
        error = ErrorDirectOutput()
        self.error_hidden_unit = error 
        self.error_output_unit = error
        pass

    def compute_activation(self, inputs, weights):
        def mul(x, y): return x * y
        def add_(x, y): return x + y
        return reduce(add_, map(mul, inputs, weights))

    def compute_derivative(self, x):
        return 1


from error import ErrorDirectOutput

class ActivatorPerceptron(Activator):
    "Perceptron activation function class"

    def __init__(self):
        error = ErrorDirectOutput()
        self.error_hidden_unit = error 
        self.error_output_unit = error
        pass

    def compute_activation(self, inputs, weights):

        value = 0
        for input, weight in zip(inputs, weights):
            value = value + input * weight

        if value > 0:   return 1
        else:           return -1

    def compute_derivative(self, x):
        return 1

from error import ErrorWeightedSum
from error import ErrorDirectOutput
import math

class ActivatorSigmoid(Activator):
    "Sigmoid activation function class"

    def __init__(self):
        self.error_hidden_unit = ErrorWeightedSum()
        self.error_output_unit = ErrorDirectOutput()
        pass

    def compute_activation(self, inputs, weights):
        def mul(x, y): return x*y
        def add_(x, y): return x+y
        def sigmoid(x): return 1 /(1 + math.exp(-x))

        value = 0
        for input, weight in zip(inputs, weights):
            value = value + input * weight

        # value = reduce(add_, map(mul, inputs, weights))
        value = sigmoid(value) 
        return value 

    def compute_derivative(self, x):
        return x *(1 - x)

