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
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

// TODO: Add support for structs with fields 
// - [T; N]
// - SMatrix<T, R, C>
// - Complex<T>

/// Derive macro for the `State` trait.
/// 
/// This macro generates implementations of:
/// - The `State` trait for accessing state elements
/// - Required arithmetic operations (Add, Sub, AddAssign, Mul<T>, Div<T>)
/// - Clone, Copy, and Debug
/// 
/// It assumes that the struct has at least one field and that all fields have the same type T,
/// where T implements the required traits.
/// 
/// # Example
/// ```rust
/// #[derive(State)]
/// struct MyState<T> {
///    x: T,
///    y: T,
/// }
/// ```
/// 
/// Note that all fields must be of type T, there can be no unused fields of a different type.
/// 
#[proc_macro_derive(State)]
pub fn derive_state(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    
    // Get the struct name
    let name = input.ident;
    
    // Extract field information
    let fields = match input.data {
        Data::Struct(data) => {
            match data.fields {
                Fields::Named(fields) => fields.named,
                Fields::Unnamed(fields) => fields.unnamed,
                Fields::Unit => panic!("Unit structs are not supported"),
            }
        },
        _ => panic!("State can only be derived for structs"),
    };
    
    // Count fields
    let field_count = fields.len();
    
    // Generate the field access expressions based on field type
    let field_accessors = fields.iter().enumerate().map(|(i, f)| {
        let field_name = match &f.ident {
            Some(ident) => quote! { self.#ident },
            None => {
                let index = syn::Index::from(i);
                quote! { self.#index }
            }
        };
        
        quote! {
            if i == #i {
                #field_name
            }
        }
    }).collect::<Vec<_>>();

    // Generate field setters based on field type
    let field_setters = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                quote! {
                    if i == #i {
                        self.#ident = value;
                        return;
                    }
                }
            },
            None => {
                let index = syn::Index::from(i);
                quote! {
                    if i == #i {
                        self.#index = value;
                        return;
                    }
                }
            }
        }
    }).collect::<Vec<_>>();

    // Generate zeros initialization based on field type
    let zeros_init = fields.iter().map(|f| {
        match &f.ident {
            Some(ident) => {
                quote! { #ident: T::zero() }
            },
            None => quote! { T::zero() }
        }
    }).collect::<Vec<_>>();

    // Generate add operation fields
    let add_fields = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                quote! { #ident: self.#ident + rhs.#ident }
            },
            None => {
                let index = syn::Index::from(i);
                quote! { self.#index + rhs.#index }
            }
        }
    }).collect::<Vec<_>>();

    // Generate sub operation fields
    let sub_fields = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                quote! { #ident: self.#ident - rhs.#ident }
            },
            None => {
                let index = syn::Index::from(i);
                quote! { self.#index - rhs.#index }
            }
        }
    }).collect::<Vec<_>>();

    // Generate add_assign operations
    let add_assign_ops = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                quote! { self.#ident += rhs.#ident; }
            },
            None => {
                let index = syn::Index::from(i);
                quote! { self.#index += rhs.#index; }
            }
        }
    }).collect::<Vec<_>>();

    // Generate mul operation fields
    let mul_fields = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                quote! { #ident: self.#ident * rhs }
            },
            None => {
                let index = syn::Index::from(i);
                quote! { self.#index * rhs }
            }
        }
    }).collect::<Vec<_>>();

    // Generate div operation fields
    let div_fields = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                quote! { #ident: self.#ident / rhs }
            },
            None => {
                let index = syn::Index::from(i);
                quote! { self.#index / rhs }
            }
        }
    }).collect::<Vec<_>>();
    
    // Generate debug field implementations based on field type
    let debug_fields = fields.iter().enumerate().map(|(i, f)| {
        match &f.ident {
            Some(ident) => {
                let ident_str = ident.to_string();
                quote! { .field(#ident_str, &self.#ident) }
            },
            None => {
                let index = syn::Index::from(i);
                let index_str = i.to_string();
                quote! { .field(#index_str, &self.#index) }
            }
        }
    }).collect::<Vec<_>>();
    
    // Generate the implementation
    let expanded = quote! {
        // Derive Clone and Copy traits directly
        impl<T: differential_equations::traits::Real> Clone for #name<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        
        impl<T: differential_equations::traits::Real> Copy for #name<T> {}

        // Implement Debug trait
        impl<T: differential_equations::traits::Real + std::fmt::Debug> std::fmt::Debug for #name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!(#name))
                    #(#debug_fields)*
                    .finish()
            }
        }

        // Implement the State trait
        impl<T: differential_equations::traits::Real> differential_equations::traits::State<T> for #name<T> {
            fn len(&self) -> usize {
                #field_count
            }
            
            fn get(&self, i: usize) -> T {
                #(#field_accessors else)* {
                    panic!("Index out of bounds")
                }
            }
            
            fn set(&mut self, i: usize, value: T) {
                #(#field_setters)*
                panic!("Index out of bounds");
            }
            
            fn zeros() -> Self {
                Self {
                    #(#zeros_init),*
                }
            }
        }

        // Implement Add
        impl<T: differential_equations::traits::Real> std::ops::Add for #name<T> {
            type Output = Self;
            
            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    #(#add_fields),*
                }
            }
        }

        // Implement Sub
        impl<T: differential_equations::traits::Real> std::ops::Sub for #name<T> {
            type Output = Self;
            
            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    #(#sub_fields),*
                }
            }
        }

        // Implement AddAssign
        impl<T: differential_equations::traits::Real> std::ops::AddAssign for #name<T> {
            fn add_assign(&mut self, rhs: Self) {
                #(#add_assign_ops)*
            }
        }

        // Implement Mul with scalar
        impl<T: differential_equations::traits::Real> std::ops::Mul<T> for #name<T> {
            type Output = Self;
            
            fn mul(self, rhs: T) -> Self::Output {
                Self {
                    #(#mul_fields),*
                }
            }
        }

        // Implement Div with scalar
        impl<T: differential_equations::traits::Real> std::ops::Div<T> for #name<T> {
            type Output = Self;
            
            fn div(self, rhs: T) -> Self::Output {
                Self {
                    #(#div_fields),*
                }
            }
        }
    };
    
    // Return the generated implementation
    TokenStream::from(expanded)
}