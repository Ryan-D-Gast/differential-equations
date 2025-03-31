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
        """Check if a Fortran implementation exists for the problem"""
        problem_dir = self.base_dir / "problems" / problem / "fortran"
        return problem_dir.exists() and (problem_dir / "Makefile").exists()
    
    def run_benchmark(self, problem, repeats, settings):
        """Run the Fortran benchmark for the problem"""
        problem_dir = self.base_dir / "problems" / problem / "fortran"
        
        # Check if gfortran is available
        if not shutil.which("gfortran"):
            print("Error: gfortran compiler not found in PATH")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": "gfortran compiler not found"
            }
        
        # Detect implementations from the Makefile
        implementations = self._get_implementations(problem_dir)
        if not implementations:
            print(f"No Fortran implementations found for {problem}")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": "No implementations found"
            }
        
        # Compile all implementations
        try:
            # Create target directory if needed
            target_dir = problem_dir / "target"
            target_dir.mkdir(exist_ok=True)
            
            # Run make to build all implementations
            subprocess.run(["make", "all"], cwd=problem_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error compiling Fortran code: {e}")
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": f"Compilation failed: {str(e)}"
            }
        
        results = {}
        
        # Prepare command line arguments from settings
        arg_list = []
        for key, value in settings.items():
            if isinstance(value, list):
                # For list values like y0, add each element separately
                for item in value:
                    arg_list.append(str(item))
            else:
                arg_list.append(str(value))
        
        # Run benchmarks for each implementation
        for impl_name in implementations:
            log_file = self.logs_dir / f"{problem}_fortran_{impl_name}_{self.timestamp}.log"
            
            # Determine executable path
            executable = problem_dir / "target" / impl_name
            if platform.system() == "Windows" and not str(executable).endswith(".exe"):
                executable = Path(f"{executable}.exe")
            
            if not executable.exists():
                print(f"Warning: Executable {executable} not found after compilation")
                continue
            
            # Run the benchmark
            times = []
            solution = None
            
            # Create command with arguments
            cmd = [str(executable)] + arg_list
            
            # Warmup run
            try:
                subprocess.run(cmd, cwd=problem_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                print(f"Warning: Fortran warmup run for {impl_name} failed. Continuing with benchmarks.")
            
            # Benchmark runs
            with open(log_file, 'w') as log:
                for i in range(repeats):
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=problem_dir,
                            text=True,
                            capture_output=True,
                            check=True
                        )
                        log.write(result.stdout)
                        
                        # Extract time and solution using regex
                        # Pattern matches: [value1, value2] time
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
                print(f"Warning: No valid timing data found for Fortran {impl_name} implementation")
                impl_results = {
                    "times": [],
                    "solution": None,
                    "average_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "stddev": 0,
                    "error": "No valid timing data found"
                }
            else:
                # Calculate statistics
                impl_results = {
                    "times": times,
                    "solution": solution,
                    "average_time": statistics.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "stddev": statistics.stdev(times) if len(times) > 1 else 0
                }
            
            results[impl_name] = impl_results
        
        # Select the best implementation as the main result
        if not results:
            return {
                "times": [],
                "solution": None,
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "stddev": 0,
                "error": "All implementations failed"
            }
        
        # Find implementation with fastest average time
        best_impl = min(results.items(), key=lambda x: float('inf') if x[1]["average_time"] == 0 else x[1]["average_time"])
        best_results = best_impl[1]
        best_results["implementations"] = {
            impl: {
                "average_time": data["average_time"],
                "min_time": data["min_time"],
                "max_time": data["max_time"],
                "stddev": data["stddev"],
                "solution": data["solution"]
            } for impl, data in results.items()
        }
        best_results["best_implementation"] = best_impl[0]
        
        return best_results
    
    def _get_implementations(self, problem_dir):
        """Extract implementation names from the Makefile"""
        makefile_path = problem_dir / "Makefile"
        implementations = []
        
        try:
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()
            
            # Look for target definitions not starting with a dot or containing special chars
            # This regex finds targets that are defined with a colon and don't start with "."
            targets = re.findall(r'^([a-zA-Z][a-zA-Z0-9_-]+):', makefile_content, re.MULTILINE)
            
            # Filter out common utility targets
            exclude_targets = {'all', 'clean', 'target', 'run-jw', 'run-hw', '.PHONY'}
            implementations = [t for t in targets if t not in exclude_targets and not t.startswith('.')]
        except Exception as e:
            print(f"Error parsing Makefile: {e}")
        
        return implementations