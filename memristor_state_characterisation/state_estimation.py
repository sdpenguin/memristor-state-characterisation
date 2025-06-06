import numpy as np
import math

def get_state(v, i, model_params, return_var=False):
    
    alpha1, alpha2, beta1, beta2, Gp = model_params

    # Estimation of state, x, for proposed model
    Id = alpha1 * (np.exp(beta1 * v) - 1) + alpha2 * (1 - np.exp(-beta2*v))
    x = i / (Gp*v + Id)

    # Partial derivatives of the state estimate with respect to v and i
    g_v = (-i*(Gp + alpha1*beta1*np.exp(beta1*v) + alpha2*beta2*np.exp(-beta2*v))/((Gp*v + alpha1*(np.exp(beta1*v)-1) + alpha2*(1-np.exp(-beta2*v)))**2))
    g_i = (1/((Gp*v + alpha1*(np.exp(beta1*v)-1) + alpha2*(1-np.exp(-beta2*v)))))

    # The uncertainty (variance in the estimate) is the square of the partial derivative with respect to the current
    uncertainty = g_i**2

    if return_var:
        return x, uncertainty
    else:
        return x

def get_states(voltage, current, model_params, return_var=False, quiet=False, eps=0.3):
    states = []
    uncertainties = []
    unidentifiable_states = []
    max_voltage = np.max(np.abs(voltage))
    max_current = np.max(np.abs(current))
    for i in range(len(voltage)):
        # NOTE: The gradient w.r.t. the voltage is only sensitive to changes in the value of i if i is close to 0, not so much for v if v is close to 0, hence why we don't need to put that in
        if math.isnan(voltage[i]) or math.isnan(current[i]) or current[i] == 0 or voltage[i] == 0 or abs(current[i]) < max_current*eps or abs(voltage[i]) < max_voltage*eps:
            unidentifiable_states.append(i)
            if len(states) > 0:
                states.append(np.nan)
                uncertainties.append(np.nan)
            else:
                states.append(np.nan)
                uncertainties.append(np.nan)
        else:
            state, uncertainty = get_state(voltage[i], current[i], model_params, return_var=True)
            states.append(state)
            uncertainties.append(uncertainty)
    if not quiet:
        print('Values could not be determined for {} of states: {}'.format(str(len(unidentifiable_states)) + '/' + str(len(states) + len(unidentifiable_states)), unidentifiable_states))
    if return_var:
        return np.array(states), np.array(uncertainties)
    else:
        return np.array(states)

def get_state_estimate(voltage, current, model_params, quiet=True, eps=0.01, return_uncertainty=False):
    states, uncertainties = get_states(voltage, current, model_params, return_var=True, quiet=quiet, eps=eps)
    no_nan_values = ~np.isnan(states) * ~np.isnan(uncertainties)
    states = states[no_nan_values]
    uncertainties = uncertainties[no_nan_values]
    state_estimate = np.sum(states * (1/uncertainties) / (np.sum(1/uncertainties)))
    total_uncertainty = np.sum(uncertainties * ((1/uncertainties)/(np.sum(1/uncertainties)))**2)
    # state_estimate = np.mean(states)
    if not return_uncertainty:
        return state_estimate
    else:
        return state_estimate, total_uncertainty
