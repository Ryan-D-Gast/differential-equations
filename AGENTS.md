# AGENTS.md - Contribution Guidelines for the `differential-equations` Repository

This document provides comprehensive guidelines for maintaining high-quality code and effective collaboration within the `differential-equations` project. These rules MUST be followed by all AI coding agents and contributors.

## Your Core Principles

All code you write MUST be fully optimized. "Fully optimized" includes:

- **Algorithmic Efficiency:** Maximizing big-O efficiency for memory and runtime.
- **Modern Hardware:** Using parallelization (`rayon`) and SIMD where appropriate.
- **Code Reuse:** Following DRY (Don't Repeat Yourself) principles and maximizing idiomatic Rust patterns.
- **Zero Technical Debt:** No extra code beyond what is necessary. If a small, well-maintained crate can significantly reduce implementation effort while maintaining optimal performance, ALWAYS use it.
- **Maintainability:** Prioritize clarity and long-term maintenance over "clever" or obscure solutions.

## Git & Version Control

### Conventional Commits

All commit messages MUST follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. Do NOT use emojis in commit messages, pull request titles, or descriptions.

**Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Allowed Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes to the build system or external dependencies
- `ci`: Changes to CI configuration
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

**Scope:** Recommended scopes include `ode`, `sde`, `dde`, `dae`, `linalg`, `docs`, or `derive`.

### Pull Requests
- Use the conventional commit format for the PR title.
- Provide a clear, concise description in the PR body.
- Link issues using "Closes #<issue_number>".
- Ensure all tests pass before requesting a review.

## Architecture & Codebase Workflows

### 1. API Changes
- **Documentation:** If you modify the public API, you MUST update the corresponding documentation in `docs/` and `README.md`.
- **Examples:** Changes must be reflected in the `examples/` directory. Ensure all examples compile and run (`cargo check --examples`). Many examples generate `.svg` visualizations; ensure these processes remain intact.

### 2. Testing & Benchmarks
- **Tests:** New features and bug fixes must include tests. Add unit tests inline or integration tests in `tests/` (categorized by `ode/`, `dde/`, etc.).
- **Benchmarks:** If your change affects performance, update or run benchmarks in `benches/`.
  - **NEVER** run benchmarks in parallel (they will compete for resources).
  - **NEVER** run with `target-cpu=native` or custom `RUSTFLAGS` unless specifically requested.
  - Ensure comparisons are apples-to-apples.

### 3. Procedural Macros (`derive/` crate)
- Evaluate if changes to mathematical structs/traits require updates to the code generation logic in `derive/src/`.
- Verify macro tests in `derive/tests/` (`cargo test -p derive`).

## Preferred Tools & Ecosystem

- **Management:** Use `cargo` for building and dependency management.
- **Feedback:** Use `indicatif` for progress bars in long-running operations.
- **Serialization:** Use `serde` and `serde_json`.
- **TUI:** Use `ratatui` and `crossterm` with intuitive mouse controls and scroll offset handling.
- **Web/API:** Use `axum` for servers. Keep handlers async, use layered extractors, and offload CPU-bound work to `tokio::task::spawn_blocking`.
- **Logging:** Use `tracing::error!` or `log::error!` instead of `println!`.
- **Data:** Use `polars` for tabular data. Never ingest more than 10 rows at a time into the context.
- **Python Integration:** Use `maturin` and `uv`. Always use a `.venv` (added to `.gitignore`).

## Code Style & Conventions

- **Naming:** `snake_case` for functions/variables/modules, `PascalCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- **Length:** Limit lines to 100 characters (`rustfmt` default).
- **Nuance:** Assume the user is a Python expert but a Rust novice; add comments for Rust-specific nuances (lifetimes, ownership, etc.).
- **Conciseness:** Avoid tautological or redundant comments.
- **Documentation:** `///` for public items. Include `Arguments`, `Returns`, `Errors`, and `Examples` sections for complex logic.

## Type System & Error Handling

- **Safety:** **NEVER** use `.unwrap()` in library code. Use `.expect()` only for invariant violations with descriptive messages.
- **Types:** Leverage the type system to prevent bugs (Newtypes, Enums).
- **Errors:** Use `thiserror` for library errors and `anyhow` for applications. Propagate with `?`.
- **Memory:** Avoid unnecessary allocations. Prefer `&str` over `String` or use `Cow` when ownership is conditional. Use `Vec::with_capacity()` when size is known.

## Testing Standards

- **Coverage:** Write unit tests for all new functions and types.
- **Isolation:** Mock external dependencies (file systems, etc.).
- **Pattern:** Use Arrange-Act-Assert.
- **CI/CD:** Use `#[cfg(test)]` modules and verify with `cargo test`.

## Concurrency & Security

- **Async:** Use `tokio` for async runtimes.
- **Parallelism:** Use `rayon` for CPU-bound tasks.
- **Safety:** Use `RwLock` or lock-free alternatives over `Mutex` where appropriate.
- **Secrets:** **NEVER** store secrets or API keys in code. Use `.env` (and ensure it's in `.gitignore`) with `dotenvy`.

## Pre-Commit Checklist

- [ ] All tests pass (`cargo test`)
- [ ] No compiler warnings (`cargo build`)
- [ ] Clippy passes (`cargo clippy -- -D warnings`)
- [ ] Code is formatted (`cargo fmt --check`)
- [ ] Doc comments are present for all public items
- [ ] No commented-out code or debug statements (`println!`, `dbg!`)
- [ ] No hardcoded credentials
- [ ] (If applicable) Python package rebuilt with `maturin`
- [ ] (If applicable) WASM binary rebuilt with `wasm-pack`

---

**Remember:** Prioritize clarity and maintainability over cleverness. This is your core directive.
