"""
This script reads the csvs in the /target/tests/results/ directory and plots them to compare the results of the solvers implemented in this library.
If the script `solve_ivp.py` is run before this script the csv result will be compared against the rust implementation.

Call the script from the root of the project with `python ,/tests/python_comparison/plots.py`
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(name):
    results_dir = 'target/tests/ode/results/'
    files = [f for f in os.listdir(results_dir) if name in f]
    
    line_styles = ['-', '--', '-.', ':']
    
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(results_dir, file))
        plt.plot(df['t'], df['y0'], label=file.replace('.csv', ''), linestyle=line_styles[i % len(line_styles)])
    
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.legend(loc = 'upper left')
    plt.title(f'{name} Comparison')
    plt.savefig(f'target/tests/ode/comparison/{name}_comparison.png')
    plt.close()

def main():
    os.makedirs('target/tests/ode/comparison', exist_ok=True)
    names = [
        'exponential_growth_positive',
        'exponential_growth_negative',
        'linear_equation',
        'harmonic_oscillator',
        'logistic_equation'
    ]
    for name in names:
        plot_comparison(name)

if __name__ == "__main__":
    main()
