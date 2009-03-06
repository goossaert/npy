class error:
    """error function class"""

    def __init__(self):
        pass

    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, fct_derivative):
        return None


import itertools

class ErrorDirectOutput(error):
    "output unit weight_update function class"

    def __init__(self):
        pass

    def compute_errors(self, errors, desired_output, outputs, next_unit_weights, fct_derivative):
        errors = [] 
        for desired, compute_d in itertools.izip(desired_output, outputs):
            # for sigmoid only!!!
            errors.append(fct_derivative(compute_d) *(desired - compute_d))
            #errors.append((desired - compute_d))
        #  print 'desired', desired_output, 'compute_d', outputs
        return errors


class ErrorWeightedSum(error):
    "Weight sum error function class"

    def __init__(self):
        pass

    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, fct_derivative):
        
        # Pre-allocate the error_sum list so that we can loop on it
        error_sum = []
        for i in range(len(next_unit_weights[0])):
            error_sum.append(0)

        # Compute the error_sum values
        for nexterror, weights in itertools.izip(next_unit_errors, next_unit_weights):
            for weight, error_sum_id in itertools.izip(weights, range(len(error_sum))): 
                error_sum[ error_sum_id ] = error_sum[ error_sum_id ] + nexterror * weight 

        # Multiply by the derivative of the sigmoid function to get_ the final value
        errors = [] 
        for currenterror, output in itertools.izip(error_sum, outputs):
            errors.append(fct_derivative(output) * currenterror)
            #errors.append(currenterror)

        return errors
