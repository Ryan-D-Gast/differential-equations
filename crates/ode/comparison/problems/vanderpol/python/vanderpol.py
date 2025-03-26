import numpy as np
from scipy.integrate import solve_ivp
import time
import argparse

# Define the van der Pol oscillator system
def van_der_pol(t, y, mu):
    """
    Defines the Van der Pol oscillator system.

    Args:
        t (float): Time.
        y (numpy.ndarray): State vector [position, velocity].
        mu (float): Parameter controlling nonlinearity.

    Returns:
        numpy.ndarray: Derivative of the state vector.
    """
    y1, y2 = y
    dydt = [y2, mu * (1 - y1**2) * y2 - y1]
    return dydt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Van der Pol oscillator benchmark')
    parser.add_argument('--csv', action='store_true', help='Save solution to CSV file')
    parser.add_argument('mu', type=float, help='Parameter controlling nonlinearity')
    parser.add_argument('t0', type=float, help='Start time')
    parser.add_argument('tf', type=float, help='End time')
    parser.add_argument('y0_0', type=float, help='Initial position')
    parser.add_argument('y0_1', type=float, help='Initial velocity')
    parser.add_argument('atol', type=float, help='Absolute tolerance')
    parser.add_argument('rtol', type=float, help='Relative tolerance')
    args = parser.parse_args()
            
    mu = args.mu
    y0 = [args.y0_0, args.y0_1]
    t_span = (args.t0, args.tf)
    
    start_time = time.time()
    sol = solve_ivp(van_der_pol, t_span, y0, args=(mu,), method='DOP853', rtol=args.rtol, atol=args.atol)
    end_time = time.time()

    if args.csv:
        # Save solution to CSV file
        np.savetxt("../target/python_vanderpol.csv", np.vstack([sol.t, sol.y]).T, delimiter=",")
    
    execution_time = end_time - start_time
    
    # Print in parsable format: [y0, y1] time
    print(f"[{sol.y[0, -1]:.15e}, {sol.y[1, -1]:.15e}] {execution_time:.6}")