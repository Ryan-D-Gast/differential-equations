//! Derive macros for differential equations crate
//!
//! [![GitHub](https://img.shields.io/badge/GitHub-differential--equations-blue)](https://github.com/Ryan-D-Gast/differential-equations)
//! [![Documentation](https://docs.rs/differential-equations/badge.svg)](https://docs.rs/differential-equations)
//!
//! # License
//!
//! ```text
//! Copyright 2025 Ryan D. Gast
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KINeither express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//! ```

use proc_macro::TokenStream;
use syn::{
    Data, DeriveInput, Fields, GenericArgument, PathArguments, Type, TypeArray, TypePath,
    parse_macro_input, parse_quote,
};

mod code_generation;
mod field_analysis;
mod implementations;

use code_generation::*;
use field_analysis::analyze_field_type;
use implementations::{ImplContext, *};

/// Derive macro for the `State` trait.
///
/// This macro generates implementations of:
/// - The `State` trait for accessing state elements
/// - Required arithmetic operations (Add, Sub, AddAssign, Mul<T>, Div<T>)
/// - Clone, Copy, and Debug
///
/// It supports structs with fields of type T, [T; N], SMatrix<T, R, C>, or Complex<T> where T implements the required traits.
///
/// # Example
/// ```rust
/// use differential_equations_derive::State;
/// use nalgebra::SMatrix;
/// use num_complex::Complex;
///
/// #[derive(State)]
/// struct MyState<T> {
///    x: T,
///    y: T,
///    velocities: [T; 3],
///    transformation: SMatrix<T, 2, 2>,
///    impedance: Complex<T>,
/// }
/// ```
///
/// Fields can be:
/// - Single values of type T
/// - Arrays of type [T; N] where N is a compile-time constant
/// - Matrices of type SMatrix<T, R, C> where R and C are compile-time constants
/// - Complex numbers of type Complex<T>
///
#[proc_macro_derive(State)]
pub fn derive_state(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Get the struct name and implementation target.
    let name = input.ident;
    let mut generics = input.generics;
    let has_generic_t = generics.type_params().any(|param| param.ident == "T");

    // Extract field information
    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            Fields::Unnamed(fields) => fields.unnamed,
            Fields::Unit => panic!("Unit structs are not supported"),
        },
        _ => panic!("State can only be derived for structs"),
    };

    let (scalar_ty, zero_expr) = if has_generic_t {
        for param in generics.type_params_mut() {
            if param.ident == "T" {
                param
                    .bounds
                    .push(parse_quote!(differential_equations::traits::Real));
            }
        }
        (quote::quote! { T }, quote::quote! { T::zero() })
    } else {
        let scalar_ty = infer_scalar_type(&fields)
            .unwrap_or_else(|| panic!("State derive requires at least one scalar field"));
        (scalar_ty.clone(), quote::quote! { 0.0 as #scalar_ty })
    };

    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let context = ImplContext {
        impl_generics: quote::quote! { #impl_generics },
        type_generics: quote::quote! { #type_generics },
        where_clause: quote::quote! { #where_clause },
        scalar_ty,
        zero_expr,
    };

    // Analyze field types and calculate total element count
    let mut total_elements = 0usize;
    let mut field_info = Vec::new();

    for field in &fields {
        let field_type_info = analyze_field_type(&field.ty);
        total_elements += field_type_info.element_count();
        field_info.push(field_type_info);
    }

    // Generate all the required components
    let field_get_branches = generate_get_branches(&fields, &field_info);
    let field_set_branches = generate_set_branches(&fields, &field_info);
    let zeros_init =
        generate_zeros_init(&fields, &field_info, &context.scalar_ty, &context.zero_expr);
    let debug_fields = generate_debug_fields(&fields);

    // Generate all trait implementations
    let clone_impl = generate_clone_impl(&name, &context);
    let copy_impl = generate_copy_impl(&name, &context);
    let debug_impl = generate_debug_impl(&name, &context, &debug_fields);
    let state_impl = generate_state_impl(
        &name,
        &context,
        total_elements,
        &field_get_branches,
        &field_set_branches,
        &zeros_init,
        &fields,
    );
    let add_impl = generate_add_impl(&name, &context, &fields, &field_info);
    let sub_impl = generate_sub_impl(&name, &context, &fields, &field_info);
    let add_assign_impl = generate_add_assign_impl(&name, &context, &fields, &field_info);
    let neg_impl = generate_neg_impl(&name, &context, &fields, &field_info);
    let mul_impl = generate_mul_impl(&name, &context, &fields, &field_info);
    let div_impl = generate_div_impl(&name, &context, &fields, &field_info);

    // Combine all implementations
    let expanded = quote::quote! {
        #clone_impl
        #copy_impl
        #debug_impl
        #state_impl
        #add_impl
        #sub_impl
        #add_assign_impl
        #mul_impl
        #div_impl
    #neg_impl
    };

    // Return the generated implementation
    TokenStream::from(expanded)
}

fn infer_scalar_type(
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> Option<proc_macro2::TokenStream> {
    fields
        .iter()
        .find_map(|field| infer_scalar_from_type(&field.ty))
}

fn infer_scalar_from_type(ty: &Type) -> Option<proc_macro2::TokenStream> {
    match ty {
        Type::Array(TypeArray { elem, .. }) => infer_scalar_from_type(elem),
        Type::Path(TypePath { path, .. }) => {
            let segment = path.segments.last()?;
            let type_name = segment.ident.to_string();

            if (matches!(type_name.as_str(), "SMatrix" | "Complex")
                || field_analysis::get_nalgebra_dimensions(&type_name).is_some())
                && let PathArguments::AngleBracketed(args) = &segment.arguments
            {
                return args.args.iter().find_map(|arg| match arg {
                    GenericArgument::Type(ty) => Some(quote::quote! { #ty }),
                    _ => None,
                });
            }

            Some(quote::quote! { #ty })
        }
        _ => None,
    }
}
