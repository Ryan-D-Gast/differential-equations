//! Code generation utilities for State and accessor methods

use crate::field_analysis::{FieldTypeInfo, generate_nalgebra_zeros};
use quote::quote;
use syn::{Field, punctuated::Punctuated, token::Comma};

/// Generate get method branches for field access
pub fn generate_get_branches(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> Vec<proc_macro2::TokenStream> {
    let mut field_get_branches = Vec::new();

    for (field_idx, (field, field_type)) in fields.iter().zip(field_info).enumerate() {
        let field_name = get_field_accessor(field, field_idx);
        let field_len = generate_field_len_expr(&field_name, field_type);

        match field_type {
            FieldTypeInfo::Single => {
                field_get_branches.push(quote! {
                    if i == current_index {
                        return #field_name;
                    }
                    current_index += #field_len;
                });
            }
            FieldTypeInfo::Array { array_size } => {
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        return #field_name[offset];
                    }
                    current_index = end_index;
                });
                let _ = array_size;
            }
            FieldTypeInfo::SMatrix { rows, cols } => {
                let matrix_indices = generate_matrix_indices(quote! { offset }, *cols);
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #matrix_indices
                        return #field_name[(row, col)];
                    }
                    current_index = end_index;
                });
                let _ = rows;
            }
            FieldTypeInfo::Complex => {
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        return if offset == 0 { #field_name.re } else { #field_name.im };
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::Vec => {
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        return #field_name[offset];
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::ArrayOfSMatrix {
                array_size,
                rows,
                cols,
            } => {
                let elements_per_matrix = rows * cols;
                let matrix_offset = generate_matrix_offset(quote! { offset }, elements_per_matrix);
                let matrix_indices = generate_matrix_indices(quote! { matrix_offset }, *cols);
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        let matrix_idx = offset / #elements_per_matrix;
                        #matrix_offset
                        #matrix_indices
                        return #field_name[matrix_idx][(row, col)];
                    }
                    current_index = end_index;
                });
                let _ = array_size;
            }
            FieldTypeInfo::ArrayOfComplex { array_size } => {
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        let complex_idx = offset / 2;
                        let component = offset % 2;
                        return if component == 0 { #field_name[complex_idx].re } else { #field_name[complex_idx].im };
                    }
                    current_index = end_index;
                });
                let _ = array_size;
            }
            FieldTypeInfo::VecOfSMatrix { rows, cols } => {
                let elements_per_matrix = rows * cols;
                let matrix_offset = generate_matrix_offset(quote! { offset }, elements_per_matrix);
                let matrix_indices = generate_matrix_indices(quote! { matrix_offset }, *cols);
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        let matrix_idx = offset / #elements_per_matrix;
                        #matrix_offset
                        #matrix_indices
                        return #field_name[matrix_idx][(row, col)];
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::VecOfComplex => {
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        let complex_idx = offset / 2;
                        let component = offset % 2;
                        return if component == 0 { #field_name[complex_idx].re } else { #field_name[complex_idx].im };
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::StateContainer => {
                field_get_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        return differential_equations::traits::State::get_component(&#field_name, offset);
                    }
                    current_index = end_index;
                });
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
    let mut field_set_branches = Vec::new();

    for (field_idx, (field, field_type)) in fields.iter().zip(field_info).enumerate() {
        let field_name = get_field_accessor(field, field_idx);
        let field_len = generate_field_len_expr(&field_name, field_type);

        match field_type {
            FieldTypeInfo::Single => {
                let field_setter = get_field_setter(field, field_idx, quote! { value });
                field_set_branches.push(quote! {
                    if i == current_index {
                        #field_setter
                        return;
                    }
                    current_index += #field_len;
                });
            }
            FieldTypeInfo::Array { array_size } => {
                let array_setter = match &field.ident {
                    Some(ident) => quote! { self.#ident[offset] = value; },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { self.#index[offset] = value; }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #array_setter
                        return;
                    }
                    current_index = end_index;
                });
                let _ = array_size;
            }
            FieldTypeInfo::SMatrix { rows, cols } => {
                let matrix_indices = generate_matrix_indices(quote! { offset }, *cols);
                let matrix_setter = match &field.ident {
                    Some(ident) => quote! {
                        #matrix_indices
                        self.#ident[(row, col)] = value;
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! {
                            #matrix_indices
                            self.#index[(row, col)] = value;
                        }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #matrix_setter
                        return;
                    }
                    current_index = end_index;
                });
                let _ = rows;
            }
            FieldTypeInfo::Complex => {
                let complex_setter = match &field.ident {
                    Some(ident) => quote! {
                        if offset == 0 {
                            self.#ident.re = value;
                        } else {
                            self.#ident.im = value;
                        }
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! {
                            if offset == 0 {
                                self.#index.re = value;
                            } else {
                                self.#index.im = value;
                            }
                        }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #complex_setter
                        return;
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::Vec => {
                let vec_setter = match &field.ident {
                    Some(ident) => quote! { self.#ident[offset] = value; },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! { self.#index[offset] = value; }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #vec_setter
                        return;
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::ArrayOfSMatrix {
                array_size,
                rows,
                cols,
            } => {
                let elements_per_matrix = rows * cols;
                let matrix_offset = generate_matrix_offset(quote! { offset }, elements_per_matrix);
                let matrix_indices = generate_matrix_indices(quote! { matrix_offset }, *cols);
                let array_matrix_setter = match &field.ident {
                    Some(ident) => quote! {
                        let matrix_idx = offset / #elements_per_matrix;
                        #matrix_offset
                        #matrix_indices
                        self.#ident[matrix_idx][(row, col)] = value;
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! {
                            let matrix_idx = offset / #elements_per_matrix;
                            #matrix_offset
                            #matrix_indices
                            self.#index[matrix_idx][(row, col)] = value;
                        }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #array_matrix_setter
                        return;
                    }
                    current_index = end_index;
                });
                let _ = array_size;
            }
            FieldTypeInfo::ArrayOfComplex { array_size } => {
                let array_complex_setter = match &field.ident {
                    Some(ident) => quote! {
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
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #array_complex_setter
                        return;
                    }
                    current_index = end_index;
                });
                let _ = array_size;
            }
            FieldTypeInfo::VecOfSMatrix { rows, cols } => {
                let elements_per_matrix = rows * cols;
                let matrix_offset = generate_matrix_offset(quote! { offset }, elements_per_matrix);
                let matrix_indices = generate_matrix_indices(quote! { matrix_offset }, *cols);
                let vec_matrix_setter = match &field.ident {
                    Some(ident) => quote! {
                        let matrix_idx = offset / #elements_per_matrix;
                        #matrix_offset
                        #matrix_indices
                        self.#ident[matrix_idx][(row, col)] = value;
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! {
                            let matrix_idx = offset / #elements_per_matrix;
                            #matrix_offset
                            #matrix_indices
                            self.#index[matrix_idx][(row, col)] = value;
                        }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #vec_matrix_setter
                        return;
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::VecOfComplex => {
                let vec_complex_setter = match &field.ident {
                    Some(ident) => quote! {
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
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #vec_complex_setter
                        return;
                    }
                    current_index = end_index;
                });
            }
            FieldTypeInfo::StateContainer => {
                let state_setter = match &field.ident {
                    Some(ident) => quote! {
                        differential_equations::traits::State::set_component(&mut self.#ident, offset, value);
                    },
                    None => {
                        let index = syn::Index::from(field_idx);
                        quote! {
                            differential_equations::traits::State::set_component(&mut self.#index, offset, value);
                        }
                    }
                };
                field_set_branches.push(quote! {
                    let end_index = current_index + #field_len;
                    if (current_index..end_index).contains(&i) {
                        let offset = i - current_index;
                        #state_setter
                        return;
                    }
                    current_index = end_index;
                });
            }
        }
    }

    field_set_branches
}

/// Generate zeros initialization for fields
pub fn generate_zeros_init(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
    scalar_ty: &proc_macro2::TokenStream,
    zero: &proc_macro2::TokenStream,
) -> Vec<proc_macro2::TokenStream> {
    fields
        .iter()
        .zip(field_info)
        .map(|(field, field_type)| match &field.ident {
            Some(ident) => match field_type {
                FieldTypeInfo::Single => quote! { #ident: #zero },
                FieldTypeInfo::Array { array_size } => quote! { #ident: [#zero; #array_size] },
                FieldTypeInfo::SMatrix { rows, cols } => {
                    let zeros_expr = generate_nalgebra_zeros(scalar_ty, *rows, *cols);
                    quote! { #ident: #zeros_expr }
                }
                FieldTypeInfo::Complex => {
                    quote! { #ident: num_complex::Complex::new(#zero, #zero) }
                }
                FieldTypeInfo::Vec
                | FieldTypeInfo::VecOfSMatrix { .. }
                | FieldTypeInfo::VecOfComplex => {
                    quote! { #ident: Vec::new() }
                }
                FieldTypeInfo::StateContainer => {
                    quote! { #ident: differential_equations::traits::State::zeros() }
                }
                FieldTypeInfo::ArrayOfSMatrix {
                    array_size,
                    rows,
                    cols,
                } => {
                    let zeros_expr = generate_nalgebra_zeros(scalar_ty, *rows, *cols);
                    quote! { #ident: [#zeros_expr; #array_size] }
                }
                FieldTypeInfo::ArrayOfComplex { array_size } => {
                    quote! { #ident: [num_complex::Complex::new(#zero, #zero); #array_size] }
                }
            },
            None => match field_type {
                FieldTypeInfo::Single => quote! { #zero },
                FieldTypeInfo::Array { array_size } => quote! { [#zero; #array_size] },
                FieldTypeInfo::SMatrix { rows, cols } => {
                    generate_nalgebra_zeros(scalar_ty, *rows, *cols)
                }
                FieldTypeInfo::Complex => quote! { num_complex::Complex::new(#zero, #zero) },
                FieldTypeInfo::Vec
                | FieldTypeInfo::VecOfSMatrix { .. }
                | FieldTypeInfo::VecOfComplex => {
                    quote! { Vec::new() }
                }
                FieldTypeInfo::StateContainer => {
                    quote! { differential_equations::traits::State::zeros() }
                }
                FieldTypeInfo::ArrayOfSMatrix {
                    array_size,
                    rows,
                    cols,
                } => {
                    let zeros_expr = generate_nalgebra_zeros(scalar_ty, *rows, *cols);
                    quote! { [#zeros_expr; #array_size] }
                }
                FieldTypeInfo::ArrayOfComplex { array_size } => {
                    quote! { [num_complex::Complex::new(#zero, #zero); #array_size] }
                }
            },
        })
        .collect()
}

pub fn generate_zeros_like_init(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
    scalar_ty: &proc_macro2::TokenStream,
    zero: &proc_macro2::TokenStream,
) -> Vec<proc_macro2::TokenStream> {
    fields
        .iter()
        .enumerate()
        .zip(field_info)
        .map(|((field_idx, field), field_type)| {
            let source = get_field_accessor(field, field_idx);
            match &field.ident {
                Some(ident) => match field_type {
                    FieldTypeInfo::Vec => quote! { #ident: vec![#zero; #source.len()] },
                    FieldTypeInfo::VecOfSMatrix { rows, cols } => {
                        let zeros_expr = generate_nalgebra_zeros(scalar_ty, *rows, *cols);
                        quote! { #ident: vec![#zeros_expr; #source.len()] }
                    }
                    FieldTypeInfo::VecOfComplex => {
                        quote! { #ident: vec![num_complex::Complex::new(#zero, #zero); #source.len()] }
                    }
                    FieldTypeInfo::StateContainer => {
                        quote! { #ident: differential_equations::traits::State::zeros_like(&#source) }
                    }
                    _ => generate_zeros_init_for_field(field, field_type, scalar_ty, zero),
                },
                None => match field_type {
                    FieldTypeInfo::Vec => quote! { vec![#zero; #source.len()] },
                    FieldTypeInfo::VecOfSMatrix { rows, cols } => {
                        let zeros_expr = generate_nalgebra_zeros(scalar_ty, *rows, *cols);
                        quote! { vec![#zeros_expr; #source.len()] }
                    }
                    FieldTypeInfo::VecOfComplex => {
                        quote! { vec![num_complex::Complex::new(#zero, #zero); #source.len()] }
                    }
                    FieldTypeInfo::StateContainer => {
                        quote! { differential_equations::traits::State::zeros_like(&#source) }
                    }
                    _ => generate_zeros_init_for_field(field, field_type, scalar_ty, zero),
                },
            }
        })
        .collect()
}

pub fn generate_len_expression(
    fields: &Punctuated<Field, Comma>,
    field_info: &[FieldTypeInfo],
) -> proc_macro2::TokenStream {
    let terms =
        fields
            .iter()
            .enumerate()
            .zip(field_info)
            .map(|((field_idx, field), field_type)| {
                let field_name = get_field_accessor(field, field_idx);
                generate_field_len_expr(&field_name, field_type)
            });

    quote! { 0usize #(+ #terms)* }
}

/// Generate debug field implementations
pub fn generate_debug_fields(fields: &Punctuated<Field, Comma>) -> Vec<proc_macro2::TokenStream> {
    fields
        .iter()
        .enumerate()
        .map(|(i, f)| match &f.ident {
            Some(ident) => {
                let ident_str = ident.to_string();
                quote! { .field(#ident_str, &self.#ident) }
            }
            None => {
                let index = syn::Index::from(i);
                let index_str = i.to_string();
                quote! { .field(#index_str, &self.#index) }
            }
        })
        .collect()
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
fn get_field_setter(
    field: &Field,
    field_idx: usize,
    value_expr: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match &field.ident {
        Some(ident) => quote! { self.#ident = #value_expr; },
        None => {
            let index = syn::Index::from(field_idx);
            quote! { self.#index = #value_expr; }
        }
    }
}

fn generate_zeros_init_for_field(
    field: &Field,
    field_type: &FieldTypeInfo,
    scalar_ty: &proc_macro2::TokenStream,
    zero: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let init = match field_type {
        FieldTypeInfo::Single => quote! { #zero },
        FieldTypeInfo::Array { array_size } => quote! { [#zero; #array_size] },
        FieldTypeInfo::SMatrix { rows, cols } => generate_nalgebra_zeros(scalar_ty, *rows, *cols),
        FieldTypeInfo::Complex => quote! { num_complex::Complex::new(#zero, #zero) },
        FieldTypeInfo::Vec | FieldTypeInfo::VecOfSMatrix { .. } | FieldTypeInfo::VecOfComplex => {
            quote! { Vec::new() }
        }
        FieldTypeInfo::StateContainer => {
            quote! { differential_equations::traits::State::zeros() }
        }
        FieldTypeInfo::ArrayOfSMatrix {
            array_size,
            rows,
            cols,
        } => {
            let zeros_expr = generate_nalgebra_zeros(scalar_ty, *rows, *cols);
            quote! { [#zeros_expr; #array_size] }
        }
        FieldTypeInfo::ArrayOfComplex { array_size } => {
            quote! { [num_complex::Complex::new(#zero, #zero); #array_size] }
        }
    };

    match &field.ident {
        Some(ident) => quote! { #ident: #init },
        None => init,
    }
}

fn generate_field_len_expr(
    field: &proc_macro2::TokenStream,
    field_type: &FieldTypeInfo,
) -> proc_macro2::TokenStream {
    match field_type {
        FieldTypeInfo::Single => quote! { 1usize },
        FieldTypeInfo::Array { array_size } => quote! { #array_size },
        FieldTypeInfo::SMatrix { rows, cols } => {
            let total_elements = rows * cols;
            quote! { #total_elements }
        }
        FieldTypeInfo::Complex => quote! { 2usize },
        FieldTypeInfo::Vec => quote! { #field.len() },
        FieldTypeInfo::ArrayOfSMatrix {
            array_size,
            rows,
            cols,
        } => {
            let total_elements = array_size * rows * cols;
            quote! { #total_elements }
        }
        FieldTypeInfo::ArrayOfComplex { array_size } => {
            let total_elements = array_size * 2;
            quote! { #total_elements }
        }
        FieldTypeInfo::VecOfSMatrix { rows, cols } => {
            let elements_per_matrix = rows * cols;
            quote! { #field.len() * #elements_per_matrix }
        }
        FieldTypeInfo::VecOfComplex => quote! { #field.len() * 2usize },
        FieldTypeInfo::StateContainer => {
            quote! { differential_equations::traits::State::len(&#field) }
        }
    }
}

fn generate_matrix_indices(
    offset_expr: proc_macro2::TokenStream,
    cols: usize,
) -> proc_macro2::TokenStream {
    if cols == 1 {
        quote! {
            let row = #offset_expr;
            let col = 0usize;
        }
    } else {
        quote! {
            let row = #offset_expr / #cols;
            let col = #offset_expr % #cols;
        }
    }
}

fn generate_matrix_offset(
    offset_expr: proc_macro2::TokenStream,
    elements_per_matrix: usize,
) -> proc_macro2::TokenStream {
    if elements_per_matrix == 1 {
        quote! {
            let matrix_offset = 0usize;
        }
    } else {
        quote! {
            let matrix_offset = #offset_expr % #elements_per_matrix;
        }
    }
}
