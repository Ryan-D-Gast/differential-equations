import numpy as np
from scipy.integrate import solve_ivp
import time
import sys

# Define the Circular Restricted Three-Body Problem system
def cr3bp(t, y, mu):
    """
    Defines the Circular Restricted Three-Body Problem (CR3BP) system.

    Args:
        t (float): Time.
        y (numpy.ndarray): State vector [x, y, z, vx, vy, vz].
        mu (float): Mass parameter of the system.

    Returns:
        numpy.ndarray: Derivative of the state vector.
    """
    x, y, z, vx, vy, vz = y
    
    # Compute distances to the primary masses
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1-mu))**2 + y**2 + z**2)
    
    # Equations of motion in the rotating frame
    dydt = np.zeros(6)
    dydt[0] = vx                                 # dx/dt = vx
    dydt[1] = vy                                 # dy/dt = vy
    dydt[2] = vz                                 # dz/dt = vz
    dydt[3] = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3  # dvx/dt
    dydt[4] = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3               # dvy/dt
    dydt[5] = -(1-mu)*z/r1**3 - mu*z/r2**3                          # dvz/dt
    
    return dydt

if __name__ == "__main__":
    # Debug info to help troubleshoot
    print(f"Arguments received: {sys.argv}", file=sys.stderr)
    
    try:
        # Check for CSV flag
        save_csv = "--csv" in sys.argv
        
        # Get the numeric arguments (skip the script name and any flags)
        numeric_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
        
        if len(numeric_args) < 11:
            print(f"Not enough arguments: expected 11, got {len(numeric_args)}", file=sys.stderr)
            sys.exit(1)
        
        # Parse arguments directly
        mu = float(numeric_args[0])
        t0 = float(numeric_args[1])
        tf = float(numeric_args[2])
        y0 = [
            float(numeric_args[3]),  # x
            float(numeric_args[4]),  # y
            float(numeric_args[5]),  # z
            float(numeric_args[6]),  # vx
            float(numeric_args[7]),  # vy
            float(numeric_args[8])   # vz
        ]
        atol = float(numeric_args[9])
        rtol = float(numeric_args[10])
        
        # Print parsed values for verification
        print(f"Parsed: mu={mu}, t0={t0}, tf={tf}, y0={y0}, atol={atol}, rtol={rtol}", file=sys.stderr)
        
        t_span = (t0, tf)
        
        start_time = time.time()
        sol = solve_ivp(cr3bp, t_span, y0, args=(mu,), method='DOP853', rtol=rtol, atol=atol)
        end_time = time.time()

        if save_csv:
            import os
            os.makedirs("../target", exist_ok=True)
            np.savetxt("../target/python_cr3bp.csv", np.vstack([sol.t, sol.y]).T, delimiter=",")
        
        execution_time = end_time - start_time
        
        # Print in parsable format: [y0, y1, ..., y5] time
        final_state = sol.y[:, -1]
        result_str = ', '.join([f"{v:.15e}" for v in final_state])
        print(f"[{result_str}] {execution_time:.6f}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)