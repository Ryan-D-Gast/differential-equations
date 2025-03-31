import os
import subprocess
from pathlib import Path
import re
import statistics
import platform
import shutil

class Harness:
    def __init__(self, base_dir, logs_dir, timestamp):
        self.base_dir = base_dir
        self.logs_dir = logs_dir
        self.timestamp = timestamp
    
    def has_implementation(self, problem):
        """Check if a Rust implementation exists for the problem"""
        problem_dir = self.base_dir / "problems" / problem / "rust"
        return problem_dir.exists() and (problem_dir / "Cargo.toml").exists()
    
    def run_benchmark(self, problem, repeats, settings):
        """Run the Rust benchmark for the problem"""
        problem_dir = self.base_dir / "problems" / problem / "rust"
        log_file = self.logs_dir / f"{problem}_rust_{self.timestamp}.log"
        
        # Build the Rust code
        try:
            print(f"Building Rust implementation in {problem_dir}...")
            subprocess.run(["cargo", "build", "--release"], cwd=problem_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error building Rust code: {e}")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": f"Build failed: {str(e)}"
            }
        
        # Determine the executable path based on OS
        if platform.system() == "Windows":
            executable = problem_dir / "target" / "release" / "rust.exe"
        else:
            executable = problem_dir / "target" / "release" / "rust"
        
        # Verify executable exists
        if not executable.exists():
            print(f"Error: Executable not found at {executable}")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": f"Executable not found: {executable}"
            }
        
        # Create command line arguments from settings
        args = [str(executable)]
        
        # Add all settings values as arguments in order
        for key, value in settings.items():
            if isinstance(value, list):
                # For list values like y0, add each element separately
                for item in value:
                    args.append(str(item))
            else:
                args.append(str(value))
        
        # Run the benchmark
        times = []
        solution = None
        
        # Warmup run
        try:
            print("Running warmup...")
            subprocess.run(args, cwd=problem_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Warmup run failed: {e}")
        
        # Benchmark runs
        with open(log_file, 'w') as log:
            print(f"Running {repeats} benchmark iterations...")
            for i in range(repeats):
                try:
                    result = subprocess.run(
                        args,
                        cwd=problem_dir,
                        text=True,
                        capture_output=True,
                        check=True
                    )
                    log.write(result.stdout)
                    
                    # Extract time and solution
                    match = re.search(r'\[(.*?)\]\s+([\d\.eE\+\-]+)', result.stdout)
                    if match:
                        if solution is None:
                            solution = match.group(1)
                        times.append(float(match.group(2)))
                except subprocess.CalledProcessError as e:
                    log.write(f"Error in run {i+1}: {e}\n")
                    if e.stdout:
                        log.write(f"STDOUT: {e.stdout}\n")
                    if e.stderr:
                        log.write(f"STDERR: {e.stderr}\n")
        
        if not times:
            print(f"Warning: No valid timing data found for Rust {problem} implementation")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": "No valid timing data found"
            }
        
        # Calculate statistics
        return {
            "times": times,
            "solution": solution,
            "average_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "stddev": statistics.stdev(times) if len(times) > 1 else 0
        }