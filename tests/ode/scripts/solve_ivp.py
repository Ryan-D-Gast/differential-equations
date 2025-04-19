"""
This script was used to generate what the expected results should be for the solvers implemented in this library.

Call the script from the root of the project with `python ./tests/python_comparison/solve_ivp.py`
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
import pandas as pd

# Define the differential equations
def exponential_growth(t, y):
    return 1 * y

def linear_equation(t, y):
    a, b = 1.0, 1.0
    return a + b * y

def harmonic_oscillator(t, y):
    dydt = [0, 0]
    dydt[0] = y[1]
    dydt[1] = -1.0 * y[0]
    return dydt

def logistic_equation(t, y):
    k, m = 1.0, 10.0
    return k * y * (1.0 - y / m)

# Options for the solver
options = {'method': 'DOP853', 'rtol': 1e-12, 'atol': 1e-12}

# Initial conditions and time span
initial_conditions = {
    'exponential_growth_positive': ([1.0], (0, 10)),
    'exponential_growth_negative': ([22026.46579479], (0, -10)),
    'linear_equation': ([1.0], (0, 10)),
    'harmonic_oscillator': ([1.0, 0.0], (0, 10)),
    'logistic_equation': ([0.1], (0, 10))
}

# Differential equations
equations = {
    'exponential_growth_positive': exponential_growth,
    'exponential_growth_negative': exponential_growth,
    'linear_equation': linear_equation,
    'harmonic_oscillator': harmonic_oscillator,
    'logistic_equation': logistic_equation
}

# Create results directory if it doesn't exist
os.makedirs('target/tests/ode/results', exist_ok=True)

# Solve and save the results for each system
for name, (y0, t_span) in initial_conditions.items():
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    solution = solve_ivp(equations[name], t_span, y0, t_eval=t_eval, **options)
    df = pd.DataFrame({'t': solution.t})
    for i in range(solution.y.shape[0]):
        df[f'y{i}'] = solution.y[i]
    df.to_csv(f'target/tests/ode/results/{name}_dop853_python.csv', index=False)
    print(f"{name}: {solution.y[:, -1]}")