# Contributing to Differential Equations

Thank you for your interest in contributing to the Differential Equations library! This project aims to build a comprehensive, high-performance suite of differential equation solvers for the Rust ecosystem. Your contributions are vital to help make this vision a reality.

## Project Structure

Currently the project is designed with the root crate being `differential-equations` and different modules for each unique type of differential equation. Each module contains its own set of solvers and utilities. This way testing, benchmarking, and documentation can be done independently for each module. In addition, while a module is in development it can be kept unincorporated into the main library until it is ready for release.

## Dependencies

This project is built around the `nalgebra` library for linear algebra operations. This is a well established library in the Rust ecosystem and has excellent performance and flexibility. Other dependencies should be kept to a minimum. Dependencies such as `polars` which can be integrated should be incorporated behind feature flags to keep the core library lightweight.

## Ways to Contribute

There are several ways you can help improve this library:

### Adding New Features
- **New Equation Types**: Implement solvers for additional types of differential equations:
  - Boundary Value Problems (BVPs)
  - Partial Differential Equations (PDEs)
  - Stochastic Differential Equations (SDEs)
  - Delay Differential Equations (DDEs)
- **New Solvers**: Implement additional numerical methods for existing equation types
- **Utility Functions**: Add helper functions for common differential equation operations

### Improving Existing Code
- **Performance Optimizations**: Identify and address performance bottlenecks
- **Memory Efficiency**: Reduce memory usage in existing algorithms
- **API Refinements**: Propose improvements to make the API more ergonomic

### Documentation and Examples
- **Improved Documentation**: Enhance existing documentation with better explanations
- **New Examples**: Create example code demonstrating use cases for the library
- **Tutorials**: Write tutorials explaining how to solve specific problems

### Testing and Validation
- **Additional Tests**: Write tests for edge cases and complex scenarios
- **Benchmarks**: Create benchmarks comparing different solvers and approaches
- **Validation**: Compare results against other established libraries and analytical solutions

## Getting Started

1. **Open an issue** - Discuss the planned changes before starting work
2. **Fork the repository** to your GitHub account
3. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/Ryan-D-Gast/differential-equations.git
   cd differential-equations
   ```
4. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** following the coding guidelines
6. **Run the tests** to ensure everything still works:
   ```bash
   cargo test
   ```
7. **Commit your changes** with descriptive commit messages
8. **Push to your fork** and submit a pull request

## Coding Guidelines

- Follow Rust's official style guidelines
- Write comprehensive documentation for public API items
- Include tests for new functionality
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Run `cargo fmt` and `cargo clippy` before submitting your code

## Pull Request Process

1. Update documentation to reflect any changes to the interface
2. Include tests that verify your changes
3. Request review from maintainers
4. Address any feedback provided during code review

## License

By contributing to this project, you agree that your contributions will be licensed under the project's Apache 2.0 License.