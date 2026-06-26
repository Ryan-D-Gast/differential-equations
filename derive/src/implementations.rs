//! Generated trait implementations for State derive macro

use crate::field_analysis::FieldTypeInfo;
use quote::quote;
use syn::{Field, Ident, punctuated::Punctuated, token::Comma};

#[derive(Clone)]
pub struct ImplContext {
    pub impl_generics: proc_macro2::TokenStream,
    pub type_generics: proc_macro2::TokenStream,
    pub where_clause: proc_macro2::TokenStream,
    pub scalar_ty: proc_macro2::TokenStream,
    pub zero_expr: proc_macro2::TokenStream,
    pub one_expr: proc_macro2::TokenStream,
}

pub struct StateImplCode<'a> {
    pub total_elements: proc_macro2::TokenStream,
    pub field_get_branches: &'a [proc_macro2::TokenStream],
    pub field_set_branches: &'a [proc_macro2::TokenStream],
    pub zeros_init: &'a [proc_macro2::TokenStream],
    pub zeros_like_init: &'a [proc_macro2::TokenStream],
}

/// Generate Add trait implementation
pub fn generate_add_impl(
    name: &Ident,
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let add_fields = generate_operation_fields(fields, field_info, OperationType::Add, context);
    let constructor = generate_constructor_syntax(&add_fields, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics std::ops::Add for #name #type_generics #where_clause {
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
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let sub_fields = generate_operation_fields(fields, field_info, OperationType::Sub, context);
    let constructor = generate_constructor_syntax(&sub_fields, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics std::ops::Sub for #name #type_generics #where_clause {
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
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let add_assign_ops = generate_assign_operations(fields, field_info, context);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics std::ops::AddAssign for #name #type_generics #where_clause {
            fn add_assign(&mut self, rhs: Self) {
                #(#add_assign_ops)*
            }
        }
    }
}

/// Generate Mul<T> trait implementation
pub fn generate_mul_impl(
    name: &Ident,
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let mul_fields = generate_operation_fields(fields, field_info, OperationType::Mul, context);
    let constructor = generate_constructor_syntax(&mul_fields, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        scalar_ty,
        ..
    } = context;

    quote! {
        impl #impl_generics std::ops::Mul<#scalar_ty> for #name #type_generics #where_clause {
            type Output = Self;

            fn mul(self, rhs: #scalar_ty) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate Div<T> trait implementation
pub fn generate_div_impl(
    name: &Ident,
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let div_fields = generate_operation_fields(fields, field_info, OperationType::Div, context);
    let constructor = generate_constructor_syntax(&div_fields, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        scalar_ty,
        ..
    } = context;

    quote! {
        impl #impl_generics std::ops::Div<#scalar_ty> for #name #type_generics #where_clause {
            type Output = Self;

            fn div(self, rhs: #scalar_ty) -> Self::Output {
                #constructor
            }
        }
    }
}

/// Generate Clone trait implementation
pub fn generate_clone_impl(
    name: &Ident,
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
) -> proc_macro2::TokenStream {
    let clone_fields = generate_clone_fields(fields);
    let constructor = generate_constructor_syntax(&clone_fields, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics Clone for #name #type_generics #where_clause {
            fn clone(&self) -> Self {
                #constructor
            }
        }
    }
}

/// Generate Copy trait implementation
pub fn generate_copy_impl(name: &Ident, context: &ImplContext) -> proc_macro2::TokenStream {
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics Copy for #name #type_generics #where_clause {}
    }
}

fn generate_clone_fields(fields: &Punctuated<Field, Comma>) -> Vec<proc_macro2::TokenStream> {
    fields
        .iter()
        .enumerate()
        .map(|(field_idx, field)| match &field.ident {
            Some(ident) => quote! { #ident: self.#ident.clone() },
            None => {
                let index = syn::Index::from(field_idx);
                quote! { self.#index.clone() }
            }
        })
        .collect()
}

/// Generate Debug trait implementation
pub fn generate_debug_impl(
    name: &Ident,
    context: &ImplContext,
    debug_fields: &[proc_macro2::TokenStream],
) -> proc_macro2::TokenStream {
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics std::fmt::Debug for #name #type_generics #where_clause {
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
    context: &ImplContext,
    code: StateImplCode<'_>,
    fields: &Punctuated<Field, Comma>,
) -> proc_macro2::TokenStream {
    let StateImplCode {
        total_elements,
        field_get_branches,
        field_set_branches,
        zeros_init,
        zeros_like_init,
    } = code;
    let zeros_constructor = generate_constructor_syntax(zeros_init, fields);
    let zeros_like_constructor = generate_constructor_syntax(zeros_like_init, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        scalar_ty,
        zero_expr,
        ..
    } = context;
    let zero = zero_expr;

    quote! {
        impl #impl_generics differential_equations::traits::State<#scalar_ty> for #name #type_generics #where_clause {
            fn len(&self) -> usize {
                #total_elements
            }

            fn get_component(&self, index: usize) -> #scalar_ty {
                self.__state_get(index)
            }

            fn set_component(&mut self, index: usize, value: #scalar_ty) {
                self.__state_set(index, value);
            }

            fn map_components_mut<F>(&mut self, mut f: F)
            where
                F: FnMut(usize, &mut #scalar_ty),
            {
                // This is a bit tricky for a struct with named fields if we don't have a way to get a &mut T by index.
                // However, the macro generates __state_set which we can use with __state_get.
                // For better performance, we'd need a way to get a &mut T.
                // Let's assume for now that we can use get/set for the generic implementation.
                for i in 0..self.len() {
                    let mut val = self.__state_get(i);
                    f(i, &mut val);
                    self.__state_set(i, val);
                }
            }

            fn zeros_like(&self) -> Self {
                #zeros_like_constructor
            }

            fn zeros() -> Self {
                #zeros_constructor
            }

            fn mul_add_assign(&mut self, alpha: #scalar_ty, other: &Self) {
                assert_eq!(self.len(), other.len(), "State length mismatch");
                for i in 0..self.len() {
                    let value = self.__state_get(i) + alpha * other.__state_get(i);
                    self.__state_set(i, value);
                }
            }

            fn scale_mut(&mut self, alpha: #scalar_ty) {
                for i in 0..self.len() {
                    let value = self.__state_get(i) * alpha;
                    self.__state_set(i, value);
                }
            }

            fn fill(&mut self, value: #scalar_ty) {
                for i in 0..self.len() {
                    self.__state_set(i, value);
                }
            }

            fn copy_from_state(&mut self, other: &Self) {
                for i in 0..self.len() {
                    self.__state_set(i, other.__state_get(i));
                }
            }

            fn norm_squared(&self) -> #scalar_ty {
                let mut sum = #zero;
                for i in 0..self.len() {
                    let value = self.__state_get(i);
                    sum += value * value;
                }
                sum
            }

            fn diff_norm_squared(&self, other: &Self) -> #scalar_ty {
                let mut sum = #zero;
                for i in 0..self.len() {
                    let diff = self.__state_get(i) - other.__state_get(i);
                    sum += diff * diff;
                }
                sum
            }

            fn error_norm(
                &self,
                y_new: &Self,
                err: &Self,
                atol: &differential_equations::tolerance::Tolerance<#scalar_ty>,
                rtol: &differential_equations::tolerance::Tolerance<#scalar_ty>,
            ) -> #scalar_ty {
                let mut sum = #zero;
                for i in 0..self.len() {
                    let sk = atol[i] + rtol[i] * self.__state_get(i).abs().max(y_new.__state_get(i).abs());
                    let e = err.__state_get(i) / sk;
                    sum += e * e;
                }
                sum
            }

            fn error_norm_inf(
                &self,
                y_new: &Self,
                err: &Self,
                atol: &differential_equations::tolerance::Tolerance<#scalar_ty>,
                rtol: &differential_equations::tolerance::Tolerance<#scalar_ty>,
            ) -> #scalar_ty {
                let mut max = #zero;
                for i in 0..self.len() {
                    let sk = atol[i] + rtol[i] * self.__state_get(i).abs().max(y_new.__state_get(i).abs());
                    max = max.max((err.__state_get(i) / sk).abs());
                }
                max
            }
        }

        impl #impl_generics #name #type_generics #where_clause {
            fn __state_get(&self, i: usize) -> #scalar_ty {
                let total_elements = #total_elements;
                assert!(i < total_elements, "Index out of bounds");
                let mut current_index = 0usize;
                #(#field_get_branches)*
                unreachable!()
            }

            fn __state_set(&mut self, i: usize, value: #scalar_ty) {
                let total_elements = #total_elements;
                assert!(i < total_elements, "Index out of bounds");
                let mut current_index = 0usize;
                #(#field_set_branches)*
                unreachable!();
            }
        }
    }
}

/// Enum for different operation types
#[derive(Clone, Copy)]
enum OperationType {
    Add,
    Sub,
    Mul,
    Div,
}

fn generate_vec_neg_expr(field: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote! {
        {
            let values = #field;
            let mut result = Vec::with_capacity(values.len());
            for value in values {
                result.push(-value);
            }
            result
        }
    }
}

fn generate_vec_operation_expr(
    lhs_field: proc_macro2::TokenStream,
    rhs_field: proc_macro2::TokenStream,
    scalar_rhs: proc_macro2::TokenStream,
    op_symbol: proc_macro2::TokenStream,
    op_type: OperationType,
) -> proc_macro2::TokenStream {
    match op_type {
        OperationType::Add | OperationType::Sub => quote! {
            {
                let lhs = #lhs_field;
                let rhs = #rhs_field;
                assert_eq!(lhs.len(), rhs.len(), "State length mismatch");
                let mut result = Vec::with_capacity(lhs.len());
                for (lhs_value, rhs_value) in lhs.into_iter().zip(rhs.into_iter()) {
                    result.push(lhs_value #op_symbol rhs_value);
                }
                result
            }
        },
        OperationType::Mul | OperationType::Div => quote! {
            {
                let values = #lhs_field;
                let mut result = Vec::with_capacity(values.len());
                for value in values {
                    result.push(value #op_symbol #scalar_rhs);
                }
                result
            }
        },
    }
}

fn generate_state_container_neg_expr(
    field: proc_macro2::TokenStream,
    one: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    quote! {
        {
            let mut result = #field;
            differential_equations::traits::State::scale_mut(&mut result, -(#one));
            result
        }
    }
}

fn generate_state_container_operation_expr(
    lhs_field: proc_macro2::TokenStream,
    rhs_field: proc_macro2::TokenStream,
    scalar_rhs: proc_macro2::TokenStream,
    op_type: OperationType,
    one: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match op_type {
        OperationType::Add => quote! {
            {
                let mut result = #lhs_field;
                let rhs = #rhs_field;
                differential_equations::traits::State::mul_add_assign(&mut result, #one, &rhs);
                result
            }
        },
        OperationType::Sub => quote! {
            {
                let mut result = #lhs_field;
                let rhs = #rhs_field;
                differential_equations::traits::State::mul_add_assign(&mut result, -(#one), &rhs);
                result
            }
        },
        OperationType::Mul => quote! {
            {
                let mut result = #lhs_field;
                differential_equations::traits::State::scale_mut(&mut result, #scalar_rhs);
                result
            }
        },
        OperationType::Div => quote! {
            {
                let mut result = #lhs_field;
                differential_equations::traits::State::scale_mut(&mut result, (#one) / #scalar_rhs);
                result
            }
        },
    }
}

/// Generate Neg trait implementation (unary minus)
pub fn generate_neg_impl(
    name: &Ident,
    context: &ImplContext,
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let neg_fields = generate_neg_fields(fields, field_info, context);
    let constructor = generate_constructor_syntax(&neg_fields, fields);
    let ImplContext {
        impl_generics,
        type_generics,
        where_clause,
        ..
    } = context;

    quote! {
        impl #impl_generics std::ops::Neg for #name #type_generics #where_clause {
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
    context: &ImplContext,
) -> Vec<proc_macro2::TokenStream> {
    let zero = &context.zero_expr;
    let one = &context.one_expr;

    fields
        .iter()
        .enumerate()
        .zip(field_info)
        .map(|((field_idx, field), field_type)| match &field.ident {
            Some(ident) => match field_type {
                FieldTypeInfo::Single => quote! { #ident: -self.#ident },
                FieldTypeInfo::Array { array_size } => {
                    quote! {
                        #ident: {
                            let mut result = [#zero; #array_size];
                            for i in 0..#array_size { result[i] = -self.#ident[i]; }
                            result
                        }
                    }
                }
                FieldTypeInfo::SMatrix { .. } => quote! { #ident: -self.#ident },
                FieldTypeInfo::Complex => quote! { #ident: -self.#ident },
                FieldTypeInfo::Vec
                | FieldTypeInfo::VecOfSMatrix { .. }
                | FieldTypeInfo::VecOfComplex => {
                    let expr = generate_vec_neg_expr(quote! { self.#ident });
                    quote! { #ident: #expr }
                }
                FieldTypeInfo::StateContainer => {
                    let expr = generate_state_container_neg_expr(quote! { self.#ident }, one);
                    quote! { #ident: #expr }
                }
                FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                    quote! {
                        #ident: {
                            let mut result = self.#ident;
                            for i in 0..#array_size { result[i] = -self.#ident[i]; }
                            result
                        }
                    }
                }
                FieldTypeInfo::ArrayOfComplex { array_size } => {
                    quote! {
                        #ident: {
                            let mut result = self.#ident;
                            for i in 0..#array_size { result[i] = -self.#ident[i]; }
                            result
                        }
                    }
                }
            },
            None => {
                let index = syn::Index::from(field_idx);
                match field_type {
                    FieldTypeInfo::Single => quote! { -self.#index },
                    FieldTypeInfo::Array { array_size } => {
                        quote! {
                            {
                                let mut result = [#zero; #array_size];
                                for i in 0..#array_size { result[i] = -self.#index[i]; }
                                result
                            }
                        }
                    }
                    FieldTypeInfo::SMatrix { .. } => quote! { -self.#index },
                    FieldTypeInfo::Complex => quote! { -self.#index },
                    FieldTypeInfo::Vec
                    | FieldTypeInfo::VecOfSMatrix { .. }
                    | FieldTypeInfo::VecOfComplex => generate_vec_neg_expr(quote! { self.#index }),
                    FieldTypeInfo::StateContainer => {
                        generate_state_container_neg_expr(quote! { self.#index }, one)
                    }
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        quote! {
                            {
                                let mut result = self.#index;
                                for i in 0..#array_size { result[i] = -self.#index[i]; }
                                result
                            }
                        }
                    }
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! {
                            {
                                let mut result = self.#index;
                                for i in 0..#array_size { result[i] = -self.#index[i]; }
                                result
                            }
                        }
                    }
                }
            }
        })
        .collect()
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
    context: &ImplContext,
) -> Vec<proc_macro2::TokenStream> {
    let zero = &context.zero_expr;
    let one = &context.one_expr;

    fields
        .iter()
        .enumerate()
        .zip(field_info)
        .map(|((field_idx, field), field_type)| {
            let (lhs, rhs, op_symbol) = match op_type {
                OperationType::Add => (quote! { self }, quote! { rhs }, quote! { + }),
                OperationType::Sub => (quote! { self }, quote! { rhs }, quote! { - }),
                OperationType::Mul => (quote! { self }, quote! { rhs }, quote! { * }),
                OperationType::Div => (quote! { self }, quote! { rhs }, quote! { / }),
            };

            match &field.ident {
                Some(ident) => match field_type {
                    FieldTypeInfo::Single => match op_type {
                        OperationType::Add | OperationType::Sub => {
                            quote! { #ident: #lhs.#ident #op_symbol #rhs.#ident }
                        }
                        OperationType::Mul | OperationType::Div => {
                            quote! { #ident: #lhs.#ident #op_symbol #rhs }
                        }
                    },
                    FieldTypeInfo::Array { array_size } => match op_type {
                        OperationType::Add | OperationType::Sub => {
                            quote! {
                                #ident: {
                                    let mut result = [#zero; #array_size];
                                    for i in 0..#array_size {
                                        result[i] = #lhs.#ident[i] #op_symbol #rhs.#ident[i];
                                    }
                                    result
                                }
                            }
                        }
                        OperationType::Mul | OperationType::Div => {
                            quote! {
                                #ident: {
                                    let mut result = [#zero; #array_size];
                                    for i in 0..#array_size {
                                        result[i] = #lhs.#ident[i] #op_symbol #rhs;
                                    }
                                    result
                                }
                            }
                        }
                    },
                    FieldTypeInfo::SMatrix { .. } => match op_type {
                        OperationType::Add | OperationType::Sub => {
                            quote! { #ident: #lhs.#ident #op_symbol #rhs.#ident }
                        }
                        OperationType::Mul | OperationType::Div => {
                            quote! { #ident: #lhs.#ident #op_symbol #rhs }
                        }
                    },
                    FieldTypeInfo::Complex => match op_type {
                        OperationType::Add | OperationType::Sub => {
                            quote! { #ident: #lhs.#ident #op_symbol #rhs.#ident }
                        }
                        OperationType::Mul | OperationType::Div => {
                            quote! { #ident: #lhs.#ident #op_symbol #rhs }
                        }
                    },
                    FieldTypeInfo::Vec
                    | FieldTypeInfo::VecOfSMatrix { .. }
                    | FieldTypeInfo::VecOfComplex => {
                        let expr = generate_vec_operation_expr(
                            quote! { #lhs.#ident },
                            quote! { #rhs.#ident },
                            quote! { #rhs },
                            op_symbol,
                            op_type,
                        );
                        quote! { #ident: #expr }
                    }
                    FieldTypeInfo::StateContainer => {
                        let expr = generate_state_container_operation_expr(
                            quote! { #lhs.#ident },
                            quote! { #rhs.#ident },
                            quote! { #rhs },
                            op_type,
                            one,
                        );
                        quote! { #ident: #expr }
                    }
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => match op_type {
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
                        }
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
                        }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => match op_type {
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
                        }
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
                        }
                    },
                },
                None => {
                    let index = syn::Index::from(field_idx);
                    match field_type {
                        FieldTypeInfo::Single => match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #lhs.#index #op_symbol #rhs.#index }
                            }
                            OperationType::Mul | OperationType::Div => {
                                quote! { #lhs.#index #op_symbol #rhs }
                            }
                        },
                        FieldTypeInfo::Array { array_size } => match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! {
                                    {
                                        let mut result = [#zero; #array_size];
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs.#index[i];
                                        }
                                        result
                                    }
                                }
                            }
                            OperationType::Mul | OperationType::Div => {
                                quote! {
                                    {
                                        let mut result = [#zero; #array_size];
                                        for i in 0..#array_size {
                                            result[i] = #lhs.#index[i] #op_symbol #rhs;
                                        }
                                        result
                                    }
                                }
                            }
                        },
                        FieldTypeInfo::SMatrix { .. } => match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #lhs.#index #op_symbol #rhs.#index }
                            }
                            OperationType::Mul | OperationType::Div => {
                                quote! { #lhs.#index #op_symbol #rhs }
                            }
                        },
                        FieldTypeInfo::Complex => match op_type {
                            OperationType::Add | OperationType::Sub => {
                                quote! { #lhs.#index #op_symbol #rhs.#index }
                            }
                            OperationType::Mul | OperationType::Div => {
                                quote! { #lhs.#index #op_symbol #rhs }
                            }
                        },
                        FieldTypeInfo::Vec
                        | FieldTypeInfo::VecOfSMatrix { .. }
                        | FieldTypeInfo::VecOfComplex => generate_vec_operation_expr(
                            quote! { #lhs.#index },
                            quote! { #rhs.#index },
                            quote! { #rhs },
                            op_symbol,
                            op_type,
                        ),
                        FieldTypeInfo::StateContainer => generate_state_container_operation_expr(
                            quote! { #lhs.#index },
                            quote! { #rhs.#index },
                            quote! { #rhs },
                            op_type,
                            one,
                        ),
                        FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => match op_type {
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
                            }
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
                            }
                        },
                        FieldTypeInfo::ArrayOfComplex { array_size } => match op_type {
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
                            }
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
                            }
                        },
                    }
                }
            }
        })
        .collect()
}

/// Generate AddAssign operations
fn generate_assign_operations(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
    context: &ImplContext,
) -> Vec<proc_macro2::TokenStream> {
    let one = &context.one_expr;

    fields
        .iter()
        .enumerate()
        .zip(field_info)
        .map(|((field_idx, field), field_type)| match &field.ident {
            Some(ident) => match field_type {
                FieldTypeInfo::Single => quote! { self.#ident += rhs.#ident; },
                FieldTypeInfo::Array { array_size } => {
                    quote! {
                        for i in 0..#array_size {
                            self.#ident[i] += rhs.#ident[i];
                        }
                    }
                }
                FieldTypeInfo::SMatrix { .. } => quote! { self.#ident += rhs.#ident; },
                FieldTypeInfo::Complex => quote! { self.#ident += rhs.#ident; },
                FieldTypeInfo::Vec
                | FieldTypeInfo::VecOfSMatrix { .. }
                | FieldTypeInfo::VecOfComplex => {
                    quote! {
                        assert_eq!(self.#ident.len(), rhs.#ident.len(), "State length mismatch");
                        for (lhs, rhs) in self.#ident.iter_mut().zip(rhs.#ident.into_iter()) {
                            *lhs += rhs;
                        }
                    }
                }
                FieldTypeInfo::StateContainer => {
                    quote! {
                        differential_equations::traits::State::mul_add_assign(
                            &mut self.#ident,
                            #one,
                            &rhs.#ident,
                        );
                    }
                }
                FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                    quote! {
                        for i in 0..#array_size {
                            self.#ident[i] += rhs.#ident[i];
                        }
                    }
                }
                FieldTypeInfo::ArrayOfComplex { array_size } => {
                    quote! {
                        for i in 0..#array_size {
                            self.#ident[i] += rhs.#ident[i];
                        }
                    }
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
                    }
                    FieldTypeInfo::SMatrix { .. } => quote! { self.#index += rhs.#index; },
                    FieldTypeInfo::Complex => quote! { self.#index += rhs.#index; },
                    FieldTypeInfo::Vec
                    | FieldTypeInfo::VecOfSMatrix { .. }
                    | FieldTypeInfo::VecOfComplex => {
                        quote! {
                            assert_eq!(self.#index.len(), rhs.#index.len(), "State length mismatch");
                            for (lhs, rhs) in self.#index.iter_mut().zip(rhs.#index.into_iter()) {
                                *lhs += rhs;
                            }
                        }
                    }
                    FieldTypeInfo::StateContainer => {
                        quote! {
                            differential_equations::traits::State::mul_add_assign(
                                &mut self.#index,
                                #one,
                                &rhs.#index,
                            );
                        }
                    }
                    FieldTypeInfo::ArrayOfSMatrix { array_size, .. } => {
                        quote! {
                            for i in 0..#array_size {
                                self.#index[i] += rhs.#index[i];
                            }
                        }
                    }
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! {
                            for i in 0..#array_size {
                                self.#index[i] += rhs.#index[i];
                            }
                        }
                    }
                }
            }
        })
        .collect()
}
