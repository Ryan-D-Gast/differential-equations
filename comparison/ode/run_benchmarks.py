import json
import argparse
import importlib
import datetime
import platform
import psutil  # For detailed hardware info
from pathlib import Path

class BenchmarkRunner:
    def __init__(self, config_file="config.json"):
        self.script_dir = Path(__file__).parent
        
        # Load configuration
        with open(self.script_dir / config_file) as f:
            self.config = json.load(f)
        
        # Setup output directories
        self.target_dir = self.script_dir / "target"
        self.logs_dir = self.target_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_hardware_info(self):
        """Collect system hardware information"""
        hardware_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=False),  # Physical cores
            "cpu_count_logical": psutil.cpu_count(logical=True),  # Logical cores
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_freq": None
        }
        
        # Get CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                hardware_info["cpu_freq"] = {
                    "current_mhz": cpu_freq.current,
                    "min_mhz": cpu_freq.min,
                    "max_mhz": cpu_freq.max
                }
        except:
            pass
        
        return hardware_info
        
    def run(self, problems=None, languages=None, repeats=None):
        """Run benchmarks for specified problems and languages"""
        if problems is None:
            problems = self.config["problems"]
        
        if languages is None:
            languages = self.config["languages"]
            
        if repeats is None:
            repeats = self.config["repeats"]
        
        print(f"===== Starting benchmark run {self.timestamp} =====")
        
        # Collect hardware info
        hardware_info = self.get_hardware_info()
        print(f"Running on: {hardware_info['processor']}")
        print(f"CPU cores: {hardware_info['cpu_count']} physical, {hardware_info['cpu_count_logical']} logical")
        
        print(f"Problems: {', '.join(problems)}")
        print(f"Languages: {', '.join(languages)}")
        print(f"Repeats: {repeats}")
        
        results = {}
        
        # Run benchmarks for each problem and language
        for problem in problems:
            results[problem] = {}
            print(f"\n----- Running {problem} benchmark -----")
            
            # Get problem settings from config
            problem_settings = self.config["problem_settings"].get(problem, {})
            if not problem_settings:
                print(f"No settings found for problem {problem}, skipping")
                continue
                
            for language in languages:
                try:
                    # Import the language-specific harness
                    harness_module = importlib.import_module(f"harnesses.{language}_harness")
                    harness = harness_module.Harness(self.script_dir, self.logs_dir, self.timestamp)
                    
                    # Check if implementation exists
                    if not harness.has_implementation(problem):
                        print(f"No {language} implementation found for {problem}, skipping")
                        continue
                    
                    print(f"\nRunning {language} implementation...")
                    
                    # Run the benchmark with problem settings
                    language_results = harness.run_benchmark(problem, repeats, problem_settings)
                    results[problem][language] = language_results
                    
                    print(f"{language} completed: {language_results['average_time']:.6f}s")
                    
                except ImportError:
                    print(f"Harness for {language} not found, skipping")
                except Exception as e:
                    print(f"Error running {language} benchmark for {problem}: {e}")
        
        # Save results with hardware info
        self.save_results(results, hardware_info)
        return results
    
    def save_results(self, results, hardware_info=None):
        """Save benchmark results to JSON with implementation names standardized"""
        # Create target/results directory
        results_dir = self.target_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / f"results_{self.timestamp}.json"
        
        # Clean results to remove potential circular references
        clean_results = self._clean_for_json(results)
        
        # Apply implementation naming standardization
        standardized_results = self._standardize_implementation_names(clean_results)
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "hardware_info": hardware_info,
                "config": self.config,
                "results": standardized_results
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    def _standardize_implementation_names(self, results):
        """Apply standard implementation names to the results"""
        standardized = {}
        
        for problem, problem_data in results.items():
            standardized[problem] = {}
            
            for language, language_data in problem_data.items():
                # Apply standard naming
                if language == 'python':
                    language_data['display_name'] = 'scipy'
                elif language == 'rust':
                    language_data['display_name'] = 'rgode'
                else:
                    language_data['display_name'] = language
                
                # Handle default implementation
                standardized[problem][language] = language_data
                
                # Handle multiple implementations
                if 'implementations' in language_data:
                    for impl_name, impl_data in language_data['implementations'].items():
                        # Map implementation names for display
                        display_name = impl_name
                        if 'jw_' in impl_name:
                            display_name = 'williams'
                        elif 'hw_' in impl_name:
                            display_name = 'hairer'
                        
                        impl_data['display_name'] = display_name
        
        return standardized
    
    def _clean_for_json(self, obj, seen=None):
        if seen is None:
            seen = set()
        
        # Use object ID to detect circular references
        obj_id = id(obj)
        if obj_id in seen:
            return "<circular reference>"
        
        if isinstance(obj, dict):
            seen.add(obj_id)
            return {k: self._clean_for_json(v, seen.copy()) for k, v in obj.items()}
        elif isinstance(obj, list):
            seen.add(obj_id)
            return [self._clean_for_json(item, seen.copy()) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            # Handle custom objects
            seen.add(obj_id)
            return {"<class>": obj.__class__.__name__, 
                    "data": self._clean_for_json(obj.__dict__, seen.copy())}
        else:
            # Primitive types (int, float, bool, string, None) can be returned as is
            return obj

def main():
    parser = argparse.ArgumentParser(description="Run cross-language benchmarks")
    parser.add_argument("--problems", nargs="+", help="Specific problems to run")
    parser.add_argument("--languages", nargs="+", help="Specific languages to run")
    parser.add_argument("--repeats", type=int, help="Number of repeats")
    parser.add_argument("--config", default="config.json", help="Config file")
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config)
    runner.run(args.problems, args.languages, args.repeats)

if __name__ == "__main__":
    main()