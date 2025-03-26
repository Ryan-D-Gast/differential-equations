# Plots phase space for Van der Pol oscillator, Note only Python vs Rust is plotted as Scipy's DOP853 implementation is the same as the fortran used
# But called via Python and thus slower but same results or at least very close only python vs rust plot is needed to show the correct output is produced.

import numpy as np
import matplotlib.pyplot as plt

# Load data
rust_data = np.loadtxt("target/rust_vanderpol.csv", delimiter=",", skiprows=1)
python_data = np.loadtxt("target/python_vanderpol.csv", delimiter=",")

# Create phase space plot (position vs. velocity)
plt.figure(figsize=(8, 6))
plt.plot(rust_data[:, 1], rust_data[:, 2], label="Rust", linewidth=2)
plt.plot(python_data[:, 1], python_data[:, 2], label="Python", linestyle='--', linewidth=1)

plt.xlabel("Position (x)", fontsize=12)
plt.ylabel("Velocity (y)", fontsize=12)
plt.title("Van der Pol Oscillator Phase Space", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("target/vanderpol_phase_space.png", dpi=300)
plt.show()