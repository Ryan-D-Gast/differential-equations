# Architecture

This library's core design philosophy is **trait-driven design**. Components are defined as traits, allowing for multiple implementations that share a common interface. This approach centralizes solver functionality, promoting DRY (Don't Repeat Yourself) principles and reducing potential bugs. Unlike traditional C/Fortran implementations where each solver duplicates functionalities like event handling and output control, this trait-driven design consolidates these common features. This not only minimizes code duplication but also simplifies the development of new solvers, as they only need to implement solver-specific logic.

The library's components are organized into four main categories:

- **Solver Controller**: This central component orchestrates the solving process. It utilizes a chosen numerical method to step through the problem, tracks statistics, manages events, and invokes the user-provided `solout` (solution output) function between steps. For Ordinary Differential Equations (ODEs), the `solve_ode` function serves this role. To manage the complexity of generics, it's often abstracted via structs like `ODEProblem`, which simplifies solving ODEs by implicitly assigning generic types.

- **Numerical Methods**: These components define the initialization and step functions for a given solver. Each numerical method is encapsulated in a struct that holds its state, enabling the solver controller to manage various differential solvers. Importantly, all numerical methods also implement an interpolation trait. This separation allows `solout` functions to access solver state for interpolation purposes in a controlled manner.

- **Differential Equation Definition**: This component represents the differential equation to be solved, defined by a user-implemented trait. Using a struct to define the differential equation allows for easy duplication of constants with different initial conditions. Users can choose the floating-point precision (`f32` or `f64`) and state representation (e.g., `f32`, `f64`, `nalgebra::SVector`, or custom structs with generic `T` fields via the derive `State` macro). This flexibility allows users to tailor the library to their specific needs. The respective differential equation implementor traits (e.g., `ODE`, `DDE`) also support optional event and Jacobian functions. Implementing these can define termination events or enhance the performance of implicit solvers.

- **Solution Output (`solout`)**: The `Solout` trait defines how solution points are handled and can be implemented by the user. More commonly, pre-built implementations are available and integrated into solver abstractions like `ODEProblem` and `DDEProblem`. Common implementations include:
    * `even`: Output at evenly spaced time points.
    * `dense`: Output at every solver step.
    * `crossing`: Output when a specific event function crosses zero.
    * `t_eval`: Output at user-specified time points.
    The desired output points are determined using the interpolation trait implemented by the solvers. Custom `solout` implementations also allow users to perform intermediate tasks, such as updating plots, writing to files, or calculating and storing system properties (e.g., energy).

## Event Handling

The library supports two primary types of event handling:

1.  **Differential Equation Event Function**:
    * Implemented as part of the differential equation trait.
    * Receives the current time and state, returning a control flag (e.g., continue or terminate).
    * If termination is requested, a root-finding algorithm is employed to precisely locate the event time.
    * **Use Case**: Ideal for basic event detection, such as checking if states go out of a predefined range, without needing a custom `solout` function.
    * **Limitation**: Provides only the current time and state.

2.  **`Solout` Function Control Flag**:
    * The `solout` function itself can return a control flag to stop the integration.
    * Instead of iterative root-finding for the event point, the `solout` function is expected to use the provided interpolation capabilities to log the final state if needed.
    * **Use Case**: Offers more flexibility for complex event handling and allows for custom actions upon event detection.
    * **Consideration**: Requires more implementation effort compared to the simpler event function.

In summary, the differential equation's event function is simpler for straightforward termination conditions, while the `solout` function provides greater flexibility for more complex event logic and actions.