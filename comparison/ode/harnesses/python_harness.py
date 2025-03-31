# Python Harness for running benchmarks

import subprocess
import statistics
from pathlib import Path
import re
import time

class Harness:
    def __init__(self, script_dir, logs_dir, timestamp):
        self.script_dir = script_dir
        self.logs_dir = logs_dir
        self.timestamp = timestamp
        
    def has_implementation(self, problem):
        """Check if implementation exists for this problem"""
        problem_path = self.script_dir / "problems" / problem / "python" / f"{problem}.py"
        return problem_path.exists()
        
    def run_benchmark(self, problem, repeats, settings):
        """Run the benchmark for given problem"""
        problem_path = self.script_dir / "problems" / problem / "python" / f"{problem}.py"
        
        # Format command line arguments from problem settings
        args = ["python", str(problem_path)]
        
        # Add all settings values as arguments in order
        for key, value in settings.items():
            if isinstance(value, list):
                # For list values like y0, add each element separately
                for item in value:
                    args.append(str(item))
            else:
                args.append(str(value))
        
        # Create log file
        log_file = self.logs_dir / f"{problem}_python_{self.timestamp}.log"
        
        times = []
        solution = None
        
        # Run multiple times
        for i in range(repeats):
            result = subprocess.run(
                args,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Python execution failed: {result.stderr}")
            
            # Parse output
            output = result.stdout.strip()
            
            # Extract time and solution using regex (same format as Rust/Fortran)
            # Pattern matches: [value1, value2] time
            match = re.search(r'\[(.*?)\]\s+([\d\.eE\+\-]+)', output)
            if match:
                if solution is None:
                    solution = match.group(1)
                times.append(float(match.group(2)))
            else:
                raise ValueError(f"Could not parse execution time from output: {output}")
            
            # Log the run
            with open(log_file, 'a') as f:
                f.write(f"Run {i+1}: {output}\n")
        
        if not times:
            print(f"Warning: No valid timing data found for Python {problem} implementation")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": "No valid timing data found"
            }
        
        # Calculate statistics (matching rust/fortran format)
        return {
            "times": times,
            "solution": solution,
            "average_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "stddev": statistics.stdev(times) if len(times) > 1 else 0
        }