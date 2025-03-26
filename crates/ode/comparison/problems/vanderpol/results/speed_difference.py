# Calculates the speed difference between implementations from logs created in target/logs/(rust|python|fortran)_date_time.log
# Averages the rows which are in form of [y0, y1] time and creates bar plot of the time for each implementation

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare execution times across language implementations')
parser.add_argument('--exclude', nargs='+', choices=['rust', 'python', 'fortran'], 
                    help='Languages to exclude from the comparison')
args = parser.parse_args()

# Get all the log files for each respective language implementation
logs = os.listdir("target/logs")
excluded_languages = args.exclude or []

languages_to_process = {
    'rust': 'rust' not in excluded_languages,
    'python': 'python' not in excluded_languages,
    'fortran': 'fortran' not in excluded_languages
}

rust_logs = [log for log in logs if log.startswith("rust")] if languages_to_process['rust'] else []
python_logs = [log for log in logs if log.startswith("python")] if languages_to_process['python'] else []
fortran_logs = [log for log in logs if "fortran" in log] if languages_to_process['fortran'] else []

print(f"Processing languages: {[lang for lang, include in languages_to_process.items() if include]}")

# Read the logs and calculate the average time for each implementation
rust_times = []
python_times = []
fortran_times = []

def extract_times_from_file(filename, is_fortran=False):
    """Extract execution times from log files with formatting specific to each language"""
    times = []
    try:
        with open(f"target/logs/{filename}", "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            for line in data:
                # Clean the line of any null bytes
                clean_line = line.strip().replace('\x00', '')
                
                # For Fortran files which have a specific format: [ value, value]     time
                if is_fortran and "]" in clean_line:
                    # Extract the last numeric value on the line
                    parts = clean_line.split()
                    if parts and len(parts) > 0:
                        try:
                            # The time is the last part
                            times.append(float(parts[-1]))
                        except ValueError:
                            continue
                else:
                    # For non-Fortran files, use regex to find execution times
                    time_matches = re.findall(r'[\*]?(\d+\.\d+)[\*]?', clean_line)
                    if time_matches:
                        try:
                            times.append(float(time_matches[-1]))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    
    return times

# Process each set of logs
for log in rust_logs:
    times = extract_times_from_file(log)
    if times:
        rust_times.append(np.mean(times))
        print(f"Rust log {log}: {len(times)} times, avg={np.mean(times):.6f}s")

for log in python_logs:
    times = extract_times_from_file(log)
    if times:
        python_times.append(np.mean(times))
        print(f"Python log {log}: {len(times)} times, avg={np.mean(times):.6f}s")

for log in fortran_logs:
    times = extract_times_from_file(log, is_fortran=True)
    if times:
        fortran_times.append(np.mean(times))
        print(f"Fortran log {log}: {len(times)} times, avg={np.mean(times):.6f}s")

# Calculate mean times if we have multiple runs
rust_mean = np.mean(rust_times) if rust_times else 0
python_mean = np.mean(python_times) if python_times else 0
fortran_mean = np.mean(fortran_times) if fortran_times else 0

# Create bar plot of average execution times
plt.figure(figsize=(10, 6))

# Only include languages that aren't excluded and have valid data
valid_langs = []
valid_means = []

if languages_to_process['rust'] and rust_mean > 0:
    valid_langs.append('Rust')
    valid_means.append(rust_mean)
    
if languages_to_process['python'] and python_mean > 0:
    valid_langs.append('Python')
    valid_means.append(python_mean)
    
if languages_to_process['fortran'] and fortran_mean > 0:
    valid_langs.append('Fortran')
    valid_means.append(fortran_mean)

if not valid_means:
    print("No valid timing data found!")
    exit(1)

# Color mapping
colors = {'Rust': 'tab:blue', 'Python': 'tab:orange', 'Fortran': 'tab:green'}
bar_colors = [colors[lang] for lang in valid_langs]

plt.bar(valid_langs, valid_means, color=bar_colors)

# Add values on top of bars
for i, v in enumerate(valid_means):
    plt.text(i, v + v*0.02, f"{v:.6f}s", ha='center')

# Add speedup factors relative to the slowest implementation
if len(valid_means) > 1:
    max_time = max(valid_means)
    for i, v in enumerate(valid_means):
        speedup = max_time / v
        if speedup > 1.0:  # Only show speedup for faster implementations
            plt.text(i, v/2, f"{speedup:.1f}x faster", ha='center', va='center',
                    color='white', fontweight='bold')

plt.xlabel("Implementation", fontsize=12)
plt.ylabel("Average Execution Time (s)", fontsize=12)
plt.title("Van der Pol Oscillator Average Execution Time", fontsize=14)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("target/vanderpol_average_time.png", dpi=300)
plt.show()