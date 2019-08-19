from keras.optimizers import Optimizer
from keras import backend as K


class RAdam(Optimizer):

    def __init__(self, lr, beta1=0.9, beta2=0.99, decay=0, **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr)
            self._beta1 = K.variable(beta1, dtype="float32")
            self._beta2 = K.variable(beta2, dtype="float32")
            self._max_sma_length = 2 / (1 - self._beta2)
            self._iterations = K.variable(0)
            self._decay = K.variable(decay)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self._iterations, 1)]
        first_moments = [K.zeros(K.int_shape(p), dtype=K.dtype(p))
                         for (i, p) in enumerate(params)]
        second_moments = [K.zeros(K.int_shape(p), dtype=K.dtype(p))
                          for (i, p) in enumerate(params)]

        self.weights = [self._iterations] + first_moments + second_moments
        bias_corrected_beta1 = K.pow(self._beta1, self._iterations)
        bias_corrected_beta2 = K.pow(self._beta2, self._iterations)
        for i, (curr_params, curr_grads) in enumerate(zip(params, grads)):
            # Updating moving moments

            new_first_moment = self._beta1 * first_moments[i] + (
                    1 - self._beta1) * curr_grads
            new_second_moment = self._beta2 * second_moments[i] + (
                    1 - self._beta2) * K.square(curr_grads)
            self.updates.append(K.update(first_moments[i],
                                         new_first_moment))
            self.updates.append(K.update(second_moments[i],
                                         new_second_moment))

            # Computing length of approximated SMA

            bias_corrected_moving_average = new_first_moment / (
                    1 - bias_corrected_beta1)
            sma_length = self._max_sma_length - 2 * (
                    self._iterations * bias_corrected_beta2) / (
                                 1 - bias_corrected_beta2)

            # Bias correction

            variance_rectification_term = K.sqrt(
                self._max_sma_length * (sma_length - 4) * (sma_length - 2) / (
                        sma_length * (self._max_sma_length - 4) *
                        (self._max_sma_length - 2) + K.epsilon()))
            resulting_parameters = K.switch(
                sma_length > 5, variance_rectification_term *
                bias_corrected_moving_average / K.sqrt(
                    K.epsilon() + new_second_moment / (1 -
                                                       bias_corrected_beta2)),
                bias_corrected_moving_average)
            resulting_parameters = curr_params - self.lr * resulting_parameters
            self.updates.append(K.update(curr_params, resulting_parameters))
        if self._decay != 0:
            new_lr = self.lr * (1. / (1. + self._decay * K.cast(
                self._iterations, K.dtype(self._decay))))
            self.updates.append(K.update(self.lr, new_lr))
        return self.updates

    def get_config(self):
        config = {
            "lr": float(K.get_value(self.lr)),
            "beta1": float(K.get_value(self._beta1)),
            "beta2": float(K.get_value(self._beta2)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
