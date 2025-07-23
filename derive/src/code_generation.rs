//! Code generation utilities for State and accessor methods

use quote::quote;
use syn::{Field, punctuated::Punctuated, token::Comma};
use crate::field_analysis::{FieldTypeInfo, generate_nalgebra_zeros};

/// Generate get method branches for field access
pub fn generate_get_branches(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> Vec<proc_macro2::TokenStream> {
    let mut current_index = 0usize;
    let mut field_get_branches = Vec::new();
    
    for (field_idx, (field, field_type)) in fields.iter().zip(field_info).enumerate() {
        let field_name = get_field_accessor(field, field_idx);
        
        match field_type {
            FieldTypeInfo::Single => {
                field_get_branches.push(quote! {
                    if i == #current_index {
                        return #field_name;
                    }
                });
                current_index += 1;
            },
            FieldTypeInfo::Array { array_size } => {
                let end_index = current_index + array_size;
                field_get_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        return #field_name[i - #current_index];
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::SMatrix { rows, cols } => {
                let total_elements = rows * cols;
                let end_index = current_index + total_elements;
                field_get_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        let offset = i - #current_index;
                        let row = offset / #cols;
                        let col = offset % #cols;
                        return #field_name[(row, col)];
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::Complex => {
                let end_index = current_index + 2;
                field_get_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        let offset = i - #current_index;
                        return if offset == 0 { #field_name.re } else { #field_name.im };
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::ArrayOfSMatrix { array_size, rows, cols } => {
                let total_elements = array_size * rows * cols;
                let elements_per_matrix = rows * cols;
                let end_index = current_index + total_elements;
                field_get_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        let offset = i - #current_index;
                        let matrix_idx = offset / #elements_per_matrix;
                        let matrix_offset = offset % #elements_per_matrix;
                        let row = matrix_offset / #cols;
                        let col = matrix_offset % #cols;
                        return #field_name[matrix_idx][(row, col)];
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::ArrayOfComplex { array_size } => {
                let total_elements = array_size * 2;
                let end_index = current_index + total_elements;
                field_get_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        let offset = i - #current_index;
                        let complex_idx = offset / 2;
                        let component = offset % 2;
                        return if component == 0 { #field_name[complex_idx].re } else { #field_name[complex_idx].im };
                    }
                });
                current_index = end_index;
            }
        }
    }
    
    field_get_branches
}

/// Generate set method branches for field access
pub fn generate_set_branches(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> Vec<proc_macro2::TokenStream> {
    let mut current_index = 0usize;
    let mut field_set_branches = Vec::new();
    
    for (field_idx, (field, field_type)) in fields.iter().zip(field_info).enumerate() {
        match field_type {
            FieldTypeInfo::Single => {
                let field_setter = get_field_setter(field, field_idx, quote! { value });
                field_set_branches.push(quote! {
                    if i == #current_index {
                        #field_setter
                        return;
                    }
                });
                current_index += 1;
            },
            FieldTypeInfo::Array { array_size } => {
                let end_index = current_index + array_size;
                let array_setter = match &field.ident {
                    Some(ident) => quote! { self.#ident[i - #current_index] = value; },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { self.#index[i - #current_index] = value; }
                    }
                };
                field_set_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        #array_setter
                        return;
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::SMatrix { rows, cols } => {
                let total_elements = rows * cols;
                let end_index = current_index + total_elements;
                let matrix_setter = match &field.ident {
                    Some(ident) => quote! { 
                        let offset = i - #current_index;
                        let row = offset / #cols;
                        let col = offset % #cols;
                        self.#ident[(row, col)] = value; 
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { 
                            let offset = i - #current_index;
                            let row = offset / #cols;
                            let col = offset % #cols;
                            self.#index[(row, col)] = value; 
                        }
                    }
                };
                field_set_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        #matrix_setter
                        return;
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::Complex => {
                let end_index = current_index + 2;
                let complex_setter = match &field.ident {
                    Some(ident) => quote! { 
                        let offset = i - #current_index;
                        if offset == 0 {
                            self.#ident.re = value;
                        } else {
                            self.#ident.im = value;
                        }
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { 
                            let offset = i - #current_index;
                            if offset == 0 {
                                self.#index.re = value;
                            } else {
                                self.#index.im = value;
                            }
                        }
                    }
                };
                field_set_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        #complex_setter
                        return;
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::ArrayOfSMatrix { array_size, rows, cols } => {
                let total_elements = array_size * rows * cols;
                let elements_per_matrix = rows * cols;
                let end_index = current_index + total_elements;
                let array_matrix_setter = match &field.ident {
                    Some(ident) => quote! { 
                        let offset = i - #current_index;
                        let matrix_idx = offset / #elements_per_matrix;
                        let matrix_offset = offset % #elements_per_matrix;
                        let row = matrix_offset / #cols;
                        let col = matrix_offset % #cols;
                        self.#ident[matrix_idx][(row, col)] = value; 
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { 
                            let offset = i - #current_index;
                            let matrix_idx = offset / #elements_per_matrix;
                            let matrix_offset = offset % #elements_per_matrix;
                            let row = matrix_offset / #cols;
                            let col = matrix_offset % #cols;
                            self.#index[matrix_idx][(row, col)] = value; 
                        }
                    }
                };
                field_set_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        #array_matrix_setter
                        return;
                    }
                });
                current_index = end_index;
            },
            FieldTypeInfo::ArrayOfComplex { array_size } => {
                let total_elements = array_size * 2;
                let end_index = current_index + total_elements;
                let array_complex_setter = match &field.ident {
                    Some(ident) => quote! { 
                        let offset = i - #current_index;
                        let complex_idx = offset / 2;
                        let component = offset % 2;
                        if component == 0 {
                            self.#ident[complex_idx].re = value;
                        } else {
                            self.#ident[complex_idx].im = value;
                        }
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { 
                            let offset = i - #current_index;
                            let complex_idx = offset / 2;
                            let component = offset % 2;
                            if component == 0 {
                                self.#index[complex_idx].re = value;
                            } else {
                                self.#index[complex_idx].im = value;
                            }
                        }
                    }
                };
                field_set_branches.push(quote! {
                    if i >= #current_index && i < #end_index {
                        #array_complex_setter
                        return;
                    }
                });
                current_index = end_index;
            }
        }
    }
    
    field_set_branches
}

/// Generate zeros initialization for fields
pub fn generate_zeros_init(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> Vec<proc_macro2::TokenStream> {
    fields.iter().zip(field_info).map(|(field, field_type)| {
        match &field.ident {
            Some(ident) => {
                match field_type {
                    FieldTypeInfo::Single => quote! { #ident: T::zero() },
                    FieldTypeInfo::Array { array_size } => quote! { #ident: [T::zero(); #array_size] },
                    FieldTypeInfo::SMatrix { rows, cols } => {
                        let zeros_expr = generate_nalgebra_zeros(*rows, *cols);
                        quote! { #ident: #zeros_expr }
                    },
                    FieldTypeInfo::Complex => quote! { #ident: num_complex::Complex::new(T::zero(), T::zero()) },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, rows, cols } => {
                        let zeros_expr = generate_nalgebra_zeros(*rows, *cols);
                        quote! { #ident: [#zeros_expr; #array_size] }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! { #ident: [num_complex::Complex::new(T::zero(), T::zero()); #array_size] }
                    },
                }
            },
            None => {
                match field_type {
                    FieldTypeInfo::Single => quote! { T::zero() },
                    FieldTypeInfo::Array { array_size } => quote! { [T::zero(); #array_size] },
                    FieldTypeInfo::SMatrix { rows, cols } => {
                        let zeros_expr = generate_nalgebra_zeros(*rows, *cols);
                        zeros_expr
                    },
                    FieldTypeInfo::Complex => quote! { num_complex::Complex::new(T::zero(), T::zero()) },
                    FieldTypeInfo::ArrayOfSMatrix { array_size, rows, cols } => {
                        let zeros_expr = generate_nalgebra_zeros(*rows, *cols);
                        quote! { [#zeros_expr; #array_size] }
                    },
                    FieldTypeInfo::ArrayOfComplex { array_size } => {
                        quote! { [num_complex::Complex::new(T::zero(), T::zero()); #array_size] }
                    },
                }
            }
        }
    }).collect()
}

/// Generate debug field implementations
pub fn generate_debug_fields(fields: &Punctuated<Field, Comma>) -> Vec<proc_macro2::TokenStream> {
    fields.iter().enumerate().map(|(i, f)| {
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
    }).collect()
}

/// Helper function to get field accessor expression
fn get_field_accessor(field: &Field, field_idx: usize) -> proc_macro2::TokenStream {
    match &field.ident {
        Some(ident) => quote! { self.#ident },
        None => {
            let index = syn::Index::from(field_idx);
            quote! { self.#index }
        }
    }
}

/// Helper function to generate field setter expression
fn get_field_setter(field: &Field, field_idx: usize, value_expr: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    match &field.ident {
        Some(ident) => quote! { self.#ident = #value_expr; },
        None => {
            let index = syn::Index::from(field_idx);
            quote! { self.#index = #value_expr; }
        }
    }
}
