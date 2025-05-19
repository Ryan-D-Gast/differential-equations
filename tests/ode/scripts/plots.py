"""
This script reads the csvs in the /target/tests/results/ directory and plots them to compare the results of the solvers implemented in this library.
If the script `solve_ivp.py` is run before this script the csv result will be compared against the rust implementation.

Call the script from the root of the project with `python ./tests/ode/scripts/plots.py`
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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
    os.makedirs('target/tests/ode/comparison/plots', exist_ok=True)
    plt.savefig(f'target/tests/ode/comparison/plots/{name}_comparison.png')
    plt.close()

def plot_error_vs_evals():
    """
    Create plots showing error vs number of function evaluations for each solver 
    at different tolerance levels. These plots help visualize solver efficiency.
    
    Dynamically finds and plots any CSV files with 'error_vs_evals' in their name.
    """
    statistics_dir = 'target/tests/ode/comparison/'
    
    # Check if directory exists
    if not os.path.exists(statistics_dir):
        print(f"Statistics directory {statistics_dir} not found. Run the statistics tests first.")
        return
    
    # Create directory for error plots
    os.makedirs('target/tests/ode/comparison/plots', exist_ok=True)
    
    # Find all CSV files with error_vs_evals in their name
    csv_files = [f for f in os.listdir(statistics_dir) if 'error_vs_evals' in f and f.endswith('.csv')]
    
    # If no files with new naming convention, try the old naming convention
    if not csv_files:
        csv_files = [f for f in os.listdir(statistics_dir) if 'error_vs_steps' in f and f.endswith('.csv')]
    
    if not csv_files:
        print("No error vs evals data found.")
        return
    
    # Process each CSV file
    for csv_file in csv_files:
        # Extract problem name from filename
        problem_name = csv_file.replace('error_vs_evals_', '').replace('error_vs_steps_', '').replace('.csv', '')
        # Capitalize first letter for the title
        title_problem_name = problem_name[0].upper() + problem_name[1:]
        
        # Load the data
        df = pd.read_csv(os.path.join(statistics_dir, csv_file))
        
        # Create figure with appropriate padding
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Get unique solvers
        solvers = df['NumericalMethod'].unique()
        
        # Different markers and colors for different solvers
        markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h']
        colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
        
        # Plot Error vs Function Evaluations
        for i, solver in enumerate(solvers):
            solver_data = df[df['NumericalMethod'] == solver]
            
            # Sort by evaluations
            solver_data = solver_data.sort_values('Evals')
            
            # Create a continuous line for each solver with distinct color
            ax.loglog(
                solver_data['Evals'],
                solver_data['Global Error'],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle='-',
                label=solver,
                markersize=8
            )
        
        # Set labels and title
        ax.set_xlabel('Number of Function Evaluations')
        ax.set_ylabel('Error')
        ax.set_title(f'{title_problem_name} - Error vs Function Evaluations')
        
        # Add grid
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save with consistent naming
        output_filename = f'error_vs_evals_{problem_name}.png'
        plt.savefig(os.path.join('target/tests/ode/comparison/plots/', output_filename), dpi=300, bbox_inches='tight')
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
    
    # Also plot the error vs steps data
    plot_error_vs_evals()

if __name__ == "__main__":
    main()
