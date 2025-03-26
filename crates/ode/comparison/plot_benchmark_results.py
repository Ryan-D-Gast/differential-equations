import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import re
import pandas as pd
import math

def load_benchmark_data(json_path):
    """Load benchmark data directly from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract hardware info
    hardware_info = data.get("hardware_info", {})
    
    # Extract initial conditions from problem settings
    problem_conditions = {}
    if "config" in data and "problem_settings" in data["config"]:
        problem_settings = data["config"]["problem_settings"]
        # Format initial conditions for each problem
        for problem, settings in problem_settings.items():
            y0_str = ", ".join([f"{y:.3g}" for y in settings.get("y0", [])])
            t_range = f"t = {settings.get('t0', 0)} to {settings.get('tf', 0)}"
            problem_conditions[problem] = f"Initial conditions: y(0) = [{y0_str}], {t_range}"
    
    # Convert JSON results to DataFrame
    rows = []
    
    for problem_name, problem_data in data["results"].items():
        for language, language_data in problem_data.items():
            # Skip if no timing data
            if "average_time" not in language_data:
                continue
            
            # Get display name (from standardized name if available)
            display_name = language_data.get('display_name', language)
                
            # Add main language data - convert seconds to milliseconds
            row = {
                'problem': problem_name,
                'language': language,
                'implementation': display_name,
                'average_time': language_data.get('average_time', 0) * 1000,  # Convert to ms
                'min_time': language_data.get('min_time', 0) * 1000,  # Convert to ms
                'max_time': language_data.get('max_time', 0) * 1000,  # Convert to ms
                'stddev': language_data.get('stddev', 0) * 1000  # Convert to ms
            }
            rows.append(row)
            
            # Handle multiple implementations
            if 'implementations' in language_data:
                for impl_name, impl_data in language_data['implementations'].items():
                    # Use display_name from standardized output if available
                    impl_display = impl_data.get('display_name', impl_name)
                    
                    impl_row = {
                        'problem': problem_name,
                        'language': language,
                        'implementation': impl_display,
                        'average_time': impl_data.get('average_time', 0) * 1000,  # Convert to ms
                        'min_time': impl_data.get('min_time', 0) * 1000,  # Convert to ms
                        'max_time': impl_data.get('max_time', 0) * 1000,  # Convert to ms
                        'stddev': impl_data.get('stddev', 0) * 1000  # Convert to ms
                    }
                    rows.append(impl_row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # For Fortran, filter out the default implementation
    if 'fortran' in df['language'].values:
        fortran_mask = (df['language'] == 'fortran') & (df['implementation'] == 'fortran')
        df = df[~fortran_mask]
    
    return df, hardware_info, problem_conditions

def format_hardware_info(hardware_info):
    """Format hardware information for display"""
    if not hardware_info:
        return "Hardware information not available"
    
    processor = hardware_info.get('processor', 'Unknown processor')
    # Clean up processor name to make it shorter
    processor = re.sub(r'\(R\)|\(TM\)|\bCPU\b|\bProcessor\b', '', processor).strip()
    
    cpu_count = hardware_info.get('cpu_count', 'N/A')
    
    # Get CPU frequency if available
    cpu_freq = hardware_info.get('cpu_freq', {})
    freq_text = ''
    if cpu_freq and 'max_mhz' in cpu_freq:
        freq_text = f", {cpu_freq['max_mhz']/1000:.2f} GHz"
    
    return f"{processor}{freq_text}, {cpu_count} cores"

def create_bar_graph(df, hardware_info=None, exclude=None, output_path=None, problem_conditions=None):
    """Create horizontal bar graphs by problem in separate subplots"""
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Filter out excluded implementations
    if exclude:
        exclude_mask = df['implementation'].isin(exclude)
        df = df[~exclude_mask]
    
    # Create a mapping of languages to colors
    languages = sorted(df['language'].unique())  # Sort languages for consistent ordering
    color_map = {lang: plt.cm.tab10(i % 10 / 10) for i, lang in enumerate(languages)}
    
    # Group by problem
    problems = df['problem'].unique()
    num_problems = len(problems)
    
    # Calculate subplot layout (try to make it somewhat square)
    cols = math.ceil(math.sqrt(num_problems))
    rows = math.ceil(num_problems / cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    axes = axes.flatten()  # Flatten to easily iterate
    
    # Process each problem in its own subplot
    for i, problem in enumerate(problems):
        ax = axes[i]
        problem_df = df[df['problem'] == problem].copy()
        
        # Sort by average time
        problem_df = problem_df.sort_values('average_time')
        
        # Use just the implementation name for labels
        positions = np.arange(len(problem_df))
        bar_height = 0.8
        
        # Determine the maximum time for this problem to set x-axis limit
        max_time = problem_df['average_time'].max() * 1.2  # Add 20% margin
        
        # Create bars
        for j, (idx, row) in enumerate(problem_df.iterrows()):
            # Use color based on language instead of problem
            bar_color = color_map[row['language']]
            
            # Create the horizontal bar with average time
            bar = ax.barh(j, row['average_time'], height=bar_height, 
                         color=bar_color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add implementation text inside bar if room
            if row['average_time'] > 10:  # Adjusted threshold for ms
                ax.text(row['average_time']/2, j, row['implementation'],
                       ha='center', va='center', fontsize=9, 
                       color='white')
        
        # Set y-axis tick labels - just implementation name
        ax.set_yticks(positions)
        ax.set_yticklabels(problem_df['implementation'].tolist())
        
        # Set up axis limits to make room for the time labels
        ax.set_xlim(0, max_time)
        
        # Add time labels to the right of the plot area
        for j, (idx, row) in enumerate(problem_df.iterrows()):
            # Format time with appropriate precision based on magnitude
            if row['average_time'] < 1:
                time_text = f"{row['average_time']:.3f} ms"
            else:
                time_text = f"{row['average_time']:.2f} ms"
                
            ax.text(max_time * 1.01, j, time_text,
                   va='center', ha='left', fontsize=8)
        
        # Set title and labels for this subplot
        ax.set_title(f"{problem}", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Only put x-label on bottom plots
        if i >= (rows-1) * cols:
            ax.set_xlabel('Time (milliseconds)')
            
            # Add initial conditions below x-axis label if available for this problem
            if problem_conditions and problem in problem_conditions:
                ax.text(0.5, -0.25, problem_conditions[problem], transform=ax.transAxes,
                       ha='center', va='center', fontsize=6, style='italic')
    
    # Hide any unused subplots
    for i in range(num_problems, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'rgode DOP853 Performance vs Fortran Implementations', fontsize=16, y=0.98)
    
    # Create language legend
    language_patches = [plt.Rectangle((0, 0), 1, 1, color=color_map[lang]) for lang in languages]
    fig.legend(language_patches, languages, loc='lower center', ncol=min(len(languages), 5), 
               title="Language", bbox_to_anchor=(0.5, 0.05), frameon=False)
    
    # Improve layout with more space at top and bottom
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.25)  # Further increased top and bottom margins
    
    # Add hardware info text box to the figure - position it lower
    if hardware_info:
        hw_text = format_hardware_info(hardware_info)
        plt.figtext(0.02, 0.02, hw_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add DOP853 explanation footnote in bottom right
    dop853_note = "DOP853: 8th-order Dormand-Prince Runge-Kutta method with adaptive step size control"
    plt.figtext(0.98, 0.02, dop853_note, fontsize=8, ha='right', 
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Speed comparison graph saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSON")
    parser.add_argument("json_file", help="Path to benchmark JSON results file")
    parser.add_argument("--exclude", nargs='+', help="Implementations to exclude")
    parser.add_argument("--output", "-o", help="Output image path (PNG, PDF, SVG, etc.)")
    args = parser.parse_args()
    
    # Determine output path if not specified
    output_path = args.output
    if not output_path:
        json_path = Path(args.json_file)
        output_path = json_path.with_suffix('.png')
    
    # Load data and create plot
    df, hardware_info, problem_conditions = load_benchmark_data(Path(args.json_file))
    create_bar_graph(df, hardware_info, args.exclude, output_path, problem_conditions)

if __name__ == "__main__":
    main()
