import re

with open('src/ivp/mod.rs', 'r') as f:
    text = f.read()

# ForwardSensitivityIvp definitions and impls
text = re.sub(r'pub struct ForwardSensitivityIvp<Previous, T: Real, P, SoloutType = DefaultSolout>', r'pub struct ForwardSensitivityIvp<Previous, T: Real, SoloutType = DefaultSolout>', text)
text = re.sub(r'pub fn forward_sensitivity<P>\(self\) -> ForwardSensitivityIvp<Self, T, P>', r'pub fn forward_sensitivity(self) -> ForwardSensitivityIvp<Self, T>', text)
text = re.sub(r'impl<Previous, T: Real, P, SoloutType> ForwardSensitivityIvp<Previous, T, P, SoloutType>', r'impl<Previous, T: Real, SoloutType> ForwardSensitivityIvp<Previous, T, SoloutType>', text)
text = re.sub(r'\) -> ForwardSensitivityIvp<Previous, T, P, NextSolout>', r') -> ForwardSensitivityIvp<Previous, T, NextSolout>', text)
text = re.sub(r'pub fn dense\(self, n: usize\) -> ForwardSensitivityIvp<Previous, T, P, DenseSolout>', r'pub fn dense(self, n: usize) -> ForwardSensitivityIvp<Previous, T, DenseSolout>', text)
text = re.sub(r'impl<\'a, F, T, Y, P, Method, BaseSolout, SensSolout>\n    ForwardSensitivityIvp<Ivp<OdeEq<\'a, F>, T, Y, Method, BaseSolout>, T, P, SensSolout>', r'impl<\'a, F, T, Y, Method, BaseSolout, SensSolout>\n    ForwardSensitivityIvp<Ivp<OdeEq<\'a, F>, T, Y, Method, BaseSolout>, T, SensSolout>', text)

# AdjointSensitivityIvp definitions and impls
text = re.sub(r'pub struct AdjointSensitivityIvp<\'c, Previous, P, Cost, BackwardMethod = SameMethod>', r'pub struct AdjointSensitivityIvp<\'c, Previous, Cost, BackwardMethod = SameMethod>', text)
text = re.sub(r'pub fn adjoint_sensitivity<\'c, P, Cost>\(', r'pub fn adjoint_sensitivity<\'c, Cost>(', text)
text = re.sub(r'\) -> AdjointSensitivityIvp<\'c, Self, P, Cost>', r') -> AdjointSensitivityIvp<\'c, Self, Cost>', text)
text = re.sub(r'impl<\'c, Previous, P, Cost, BackwardMethod>\n    AdjointSensitivityIvp<\'c, Previous, P, Cost, BackwardMethod>', r'impl<\'c, Previous, Cost, BackwardMethod>\n    AdjointSensitivityIvp<\'c, Previous, Cost, BackwardMethod>', text)
text = re.sub(r'\) -> AdjointSensitivityIvp<\'c, Previous, P, Cost, UseBackwardMethod<Method>>', r') -> AdjointSensitivityIvp<\'c, Previous, Cost, UseBackwardMethod<Method>>', text)
text = re.sub(r'impl<\'c, \'a, F, T, Y, P, Method, SoloutType, Cost>\n    AdjointSensitivityIvp<\'c, Ivp<OdeEq<\'a, F>, T, Y, Method, SoloutType>, P, Cost, SameMethod>', r'impl<\'c, \'a, F, T, Y, Method, SoloutType, Cost>\n    AdjointSensitivityIvp<\'c, Ivp<OdeEq<\'a, F>, T, Y, Method, SoloutType>, Cost, SameMethod>', text)
text = re.sub(r'impl<\'c, \'a, F, T, Y, P, Method, SoloutType, Cost, BackwardMethod>\n    AdjointSensitivityIvp<\n        \'c,\n        Ivp<OdeEq<\'a, F>, T, Y, Method, SoloutType>,\n        P,\n        Cost,\n        UseBackwardMethod<BackwardMethod>,\n    >', r'impl<\'c, \'a, F, T, Y, Method, SoloutType, Cost, BackwardMethod>\n    AdjointSensitivityIvp<\n        \'c,\n        Ivp<OdeEq<\'a, F>, T, Y, Method, SoloutType>,\n        Cost,\n        UseBackwardMethod<BackwardMethod>,\n    >', text)

# P bounds and types
text = re.sub(r'P: State<T>,\n    ', r'', text)
text = re.sub(r'F: VaryParameters<T, Y, P>', r'F: VaryParameters<T, Y>', text)
text = re.sub(r'Cost: AdjointCost<T, Y, P, F>', r'Cost: AdjointCost<T, Y, F>', text)
text = re.sub(r'AdjointState<T, Y, P>', r'AdjointState<T, Y, F::Params>', text)
text = re.sub(r'AdjointSolution<T, Y, P>', r'AdjointSolution<T, Y, F::Params>', text)

# markers
text = re.sub(r'_marker: std::marker::PhantomData<P>,', r'_marker: std::marker::PhantomData<T>,', text)
text = re.sub(r'_marker: std::marker::PhantomData,', r'_marker: std::marker::PhantomData,', text)

with open('src/ivp/mod.rs', 'w') as f:
    f.write(text)

