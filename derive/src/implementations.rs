//! Generated trait implementations for State derive macro

use quote::quote;
use syn::{Ident, Field, punctuated::Punctuated, token::Comma};
use crate::field_analysis::FieldTypeInfo;

/// Generate Add trait implementation
pub fn generate_add_impl(
    name: &Ident,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let add_fields = generate_operation_fields(fields, field_info, OperationType::Add);
    let constructor = generate_constructor_syntax(&add_fields, fields);
    
    quote! {
        impl<T: differential_equations::traits::Real> std::ops::Add for #name<T> {
            type Output = Self;
            
            fn add(self, rhs: Self) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate Sub trait implementation
pub fn generate_sub_impl(
    name: &Ident,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let sub_fields = generate_operation_fields(fields, field_info, OperationType::Sub);
    let constructor = generate_constructor_syntax(&sub_fields, fields);
    
    quote! {
        impl<T: differential_equations::traits::Real> std::ops::Sub for #name<T> {
            type Output = Self;
            
            fn sub(self, rhs: Self) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate AddAssign trait implementation
pub fn generate_add_assign_impl(
    name: &Ident,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let add_assign_ops = generate_assign_operations(fields, field_info);
    
    quote! {
        impl<T: differential_equations::traits::Real> std::ops::AddAssign for #name<T> {
            fn add_assign(&mut self, rhs: Self) {
                #(#add_assign_ops)*
            }
        }
    }
}

/// Generate Mul<T> trait implementation
pub fn generate_mul_impl(
    name: &Ident,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let mul_fields = generate_operation_fields(fields, field_info, OperationType::Mul);
    let constructor = generate_constructor_syntax(&mul_fields, fields);
    
    quote! {
        impl<T: differential_equations::traits::Real> std::ops::Mul<T> for #name<T> {
            type Output = Self;
            
            fn mul(self, rhs: T) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate Div<T> trait implementation
pub fn generate_div_impl(
    name: &Ident,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let div_fields = generate_operation_fields(fields, field_info, OperationType::Div);
    let constructor = generate_constructor_syntax(&div_fields, fields);
    
    quote! {
        impl<T: differential_equations::traits::Real> std::ops::Div<T> for #name<T> {
            type Output = Self;
            
            fn div(self, rhs: T) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate Clone trait implementation
pub fn generate_clone_impl(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        impl<T: differential_equations::traits::Real> Clone for #name<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
    }
}

/// Generate Copy trait implementation
pub fn generate_copy_impl(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        impl<T: differential_equations::traits::Real> Copy for #name<T> {}
    }
}

/// Generate Debug trait implementation
pub fn generate_debug_impl(
    name: &Ident,
    debug_fields: &[proc_macro2::TokenStream],
) -> proc_macro2::TokenStream {
    quote! {
        impl<T: differential_equations::traits::Real + std::fmt::Debug> std::fmt::Debug for #name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!(#name))
                    #(#debug_fields)*
                    .finish()
            }
        }
    }
}

/// Generate State trait implementation
pub fn generate_state_impl(
    name: &Ident,
    total_elements: usize,
    field_get_branches: &[proc_macro2::TokenStream],
    field_set_branches: &[proc_macro2::TokenStream],
    zeros_init: &[proc_macro2::TokenStream],
    fields: &Punctuated<Field, Comma>,
) -> proc_macro2::TokenStream {
    let zeros_constructor = generate_constructor_syntax(zeros_init, fields);
    
    quote! {
        impl<T: differential_equations::traits::Real> differential_equations::traits::State<T> for #name<T> {
            fn len(&self) -> usize {
                #total_elements
            }
            
            fn get(&self, i: usize) -> T {
                #(#field_get_branches)*
                panic!("Index out of bounds")
            }
            
            fn set(&mut self, i: usize, value: T) {
                #(#field_set_branches)*
                panic!("Index out of bounds");
            }
            
            fn zeros() -> Self {
                #zeros_constructor
            }
        }
    }
}

/// Enum for different operation types
enum OperationType {
    Add,
    Sub,
    Mul,
    Div,
}

/// Generate Neg trait implementation (unary minus)
pub fn generate_neg_impl(
    name: &Ident,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let neg_fields = generate_neg_fields(fields, field_info);
    let constructor = generate_constructor_syntax(&neg_fields, fields);

    quote! {
        impl<T: differential_equations::traits::Real> std::ops::Neg for #name<T> {
            type Output = Self;

            fn neg(self) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate per-field expressions for unary negation
fn generate_neg_fields(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> Vec<proc_macro2::TokenStream> {
    fields.iter().enumerate().zip(field_info).map(|((field_idx, field), field_type)| {
        match &field.ident {
            Some(ident) => {
                match field_type {
                    FieldTypeInfo::Single => quote! { #ident: -self.#ident },
                    FieldTypeInfo::Array { array_size } => {
                        quote! { 
                            #ident: {
                                let mut result = [T::zero(); #array_size];
                                for i in 0..#array_size { result[i] = -self.#ident[i]; }
                                result
                            }
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => quote! { #ident: -self.#ident },
                    FieldTypeInfo::Complex => quote! { #ident: -self.#ident },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        quote! { 
                            #ident: {
                                let mut result = self.#ident;
                                for i in 0..#array_size { result[i] = -self.#ident[i]; }
                                result
                            }
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! { 
                            #ident: {
                                let mut result = self.#ident;
                                for i in 0..#array_size { result[i] = -self.#ident[i]; }
                                result
                            }
                        }
                    },
                }
            },
            None => {
                let index = syn::Index::from(field_idx);
                match field_type {
                    FieldTypeInfo::Single => quote! { -self.#index },
                    FieldTypeInfo::Array { array_size } => {
                        quote! { 
                            {
                                let mut result = [T::zero(); #array_size];
                                for i in 0..#array_size { result[i] = -self.#index[i]; }
                                result
                            }
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => quote! { -self.#index },
                    FieldTypeInfo::Complex => quote! { -self.#index },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        quote! { 
                            {
                                let mut result = self.#index;
                                for i in 0..#array_size { result[i] = -self.#index[i]; }
                                result
                            }
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! { 
                            {
                                let mut result = self.#index;
                                for i in 0..#array_size { result[i] = -self.#index[i]; }
                                result
                            }
                        }
                    },
                }
            }
        }
    }).collect()
}

/// Helper function to determine if we're dealing with a tuple struct
fn is_tuple_struct(fields: &Punctuated<Field, Comma>) -> bool {
    fields.iter().all(|field| field.ident.is_none())
}

/// Generate constructor syntax for either named or tuple structs
fn generate_constructor_syntax(
    field_expressions: &[proc_macro2::TokenStream],
    fields: &Punctuated<Field, Comma>,
) -> proc_macro2::TokenStream {
    if is_tuple_struct(fields) {
        // Tuple struct: Self(expr1, expr2, expr3)
        quote! {
            Self(
                #(#field_expressions),*
            )
        }
    } else {
        // Named struct: Self { field1: expr1, field2: expr2 }
        quote! {
            Self {
                #(#field_expressions),*
            }
        }
    }
}

/// Generate operation fields for binary operations
fn generate_operation_fields(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
    op_type: OperationType,
) -> Vec<proc_macro2::TokenStream> {
    fields.iter().enumerate().zip(field_info).map(|((field_idx, field), field_type)| {
        let (lhs, rhs, op_symbol) = match op_type {
            OperationType::Add => (quote! { self }, quote! { rhs }, quote! { + }),
            OperationType::Sub => (quote! { self }, quote! { rhs }, quote! { - }),
            OperationType::Mul => (quote! { self }, quote! { rhs }, quote! { * }),
            OperationType::Div => (quote! { self }, quote! { rhs }, quote! { / }),
        };
        
        match &field.ident {
            Some(ident) => {
                match field_type {
                    FieldTypeInfo::Single => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #ident: #lhs.#ident #op_symbol #rhs.#ident }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { #ident: #lhs.#ident #op_symbol #rhs }
                            },
                        }
                    },
                    FieldTypeInfo::Array { array_size } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { 
                                    #ident: {
                                        let mut result = [T::zero(); #array_size];
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#ident[i] #op_symbol #rhs.#ident[i];
                                        }
                                        result
                                    }
                                }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { 
                                    #ident: {
                                        let mut result = [T::zero(); #array_size];
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#ident[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            },
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #ident: #lhs.#ident #op_symbol #rhs.#ident }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { #ident: #lhs.#ident #op_symbol #rhs }
                            },
                        }
                    },
                    FieldTypeInfo::Complex => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #ident: #lhs.#ident #op_symbol #rhs.#ident }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { #ident: #lhs.#ident #op_symbol #rhs }
                            },
                        }
                    },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { 
                                    #ident: {
                                        let mut result = #lhs.#ident;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#ident[i] #op_symbol #rhs.#ident[i];
                                        }
                                        result
                                    }
                                }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { 
                                    #ident: {
                                        let mut result = #lhs.#ident;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#ident[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            },
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { 
                                    #ident: {
                                        let mut result = #lhs.#ident;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#ident[i] #op_symbol #rhs.#ident[i];
                                        }
                                        result
                                    }
                                }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { 
                                    #ident: {
                                        let mut result = #lhs.#ident;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#ident[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            },
                        }
                    },
                }
            },
            None => {
                let index = syn::Index::from(field_idx);
                match field_type {
                    FieldTypeInfo::Single => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #lhs.#index #op_symbol #rhs.#index }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { #lhs.#index #op_symbol #rhs }
                            },
                        }
                    },
                    FieldTypeInfo::Array { array_size } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { 
                                    {
                                        let mut result = [T::zero(); #array_size];
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs.#index[i];
                                        }
                                        result
                                    }
                                }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { 
                                    {
                                        let mut result = [T::zero(); #array_size];
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            },
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #lhs.#index #op_symbol #rhs.#index }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { #lhs.#index #op_symbol #rhs }
                            },
                        }
                    },
                    FieldTypeInfo::Complex => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #lhs.#index #op_symbol #rhs.#index }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { #lhs.#index #op_symbol #rhs }
                            },
                        }
                    },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { 
                                    {
                                        let mut result = #lhs.#index;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs.#index[i];
                                        }
                                        result
                                    }
                                }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { 
                                    {
                                        let mut result = #lhs.#index;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            },
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { 
                                    {
                                        let mut result = #lhs.#index;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs.#index[i];
                                        }
                                        result
                                    }
                                }
                            },
                            OperationType::Mul | OperationType::Div => {
                                quote! { 
                                    {
                                        let mut result = #lhs.#index;
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            },
                        }
                    },
                }
            }
        }
    }).collect()
}

/// Generate AddAssign operations
fn generate_assign_operations(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> Vec<proc_macro2::TokenStream> {
    fields.iter().enumerate().zip(field_info).map(|((field_idx, field), field_type)| {
        match &field.ident {
            Some(ident) => {
                match field_type {
                    FieldTypeInfo::Single => quote! { self.#ident += rhs.#ident; },
                    FieldTypeInfo::Array { array_size } => {
                        quote! { 
                            for i in 0..#array_size {
                                self.#ident[i] += rhs.#ident[i];
                            }
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => quote! { self.#ident += rhs.#ident; },
                    FieldTypeInfo::Complex => quote! { self.#ident += rhs.#ident; },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        quote! { 
                            for i in 0..#array_size {
                                self.#ident[i] += rhs.#ident[i];
                            }
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! { 
                            for i in 0..#array_size {
                                self.#ident[i] += rhs.#ident[i];
                            }
                        }
                    },
                }
            },
            None => {
                let index = syn::Index::from(field_idx);
                match field_type {
                    FieldTypeInfo::Single => quote! { self.#index += rhs.#index; },
                    FieldTypeInfo::Array { array_size } => {
                        quote! { 
                            for i in 0..#array_size {
                                self.#index[i] += rhs.#index[i];
                            }
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => quote! { self.#index += rhs.#index; },
                    FieldTypeInfo::Complex => quote! { self.#index += rhs.#index; },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        quote! { 
                            for i in 0..#array_size {
                                self.#index[i] += rhs.#index[i];
                            }
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! { 
                            for i in 0..#array_size {
                                self.#index[i] += rhs.#index[i];
                            }
                        }
                    },
                }
            }
        }
    }).collect()
}
