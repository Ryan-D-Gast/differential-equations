# Contributing to Differential Equations

Thank you for your interest in contributing to the Differential Equations library! This project aims to build a comprehensive, high-performance suite of differential equation solvers for the Rust ecosystem. Your contributions are vital to help make this vision a reality.

## Todo List

If you're looking for ways to contribute, here are some areas where we could use help. In addition if you have ideas to create a pull request to add more.

### Miscellaneous
- Derive `State` Macro improvements
   - Add support for non `T` fields in structs
      - [ ] `[T; N]` 
      - [ ] `SMatrix<T, R, C>` 
      - [ ] `Complex<T>`

### Differential Equation Types
- Add additional support for differential equations:
   - [ ] Stochastic Delay Differential Equations (DDE)
   - Add more if you have ideas!

## Coding Guidelines

- Follow Rust's official style guidelines
- Write comprehensive documentation for public API items
- Include tests for new functionality
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Run `cargo fmt` and `cargo clippy` before submitting your code

## License

By contributing to this project, you agree that your contributions will be licensed under the project's Apache 2.0 License.