use syn::{Type, TypeArray, Expr, TypePath, PathArguments, GenericArgument};
use quote::quote;

#[derive(Clone)]
pub enum FieldTypeInfo {
    Single,
    Array { array_size: usize },
    SMatrix { rows: usize, cols: usize },
    Complex,
    ArrayOfSMatrix { array_size: usize, rows: usize, cols: usize },
    ArrayOfComplex { array_size: usize },
}

impl FieldTypeInfo {
    pub fn element_count(&self) -> usize {
        match self {
            FieldTypeInfo::Single => 1,
            FieldTypeInfo::Array { array_size } => *array_size,
            FieldTypeInfo::SMatrix { rows, cols } => rows * cols,
            FieldTypeInfo::Complex => 2,
            FieldTypeInfo::ArrayOfSMatrix { array_size, rows, cols } => array_size * rows * cols,
            FieldTypeInfo::ArrayOfComplex { array_size } => array_size * 2,
        }
    }
}

pub fn analyze_field_type(ty: &Type) -> FieldTypeInfo {
    match ty {
        Type::Array(TypeArray { elem, len, .. }) => {
            // Try to extract the array size
            let array_size = match len {
                Expr::Lit(lit) => {
                    match &lit.lit {
                        syn::Lit::Int(int_lit) => {
                            int_lit.base10_parse::<usize>().unwrap_or_else(|_| {
                                panic!("Array size must be a positive integer")
                            })
                        },
                        _ => panic!("Array size must be an integer literal"),
                    }
                },
                _ => panic!("Array size must be a compile-time constant"),
            };
            
            // Analyze the element type to see if it's a complex type
            match analyze_field_type(elem) {
                FieldTypeInfo::SMatrix { rows, cols } => {
                    FieldTypeInfo::ArrayOfSMatrix { array_size, rows, cols }
                },
                FieldTypeInfo::Complex => {
                    FieldTypeInfo::ArrayOfComplex { array_size }
                },
                _ => FieldTypeInfo::Array { array_size }
            }
        },
        Type::Path(TypePath { path, .. }) => {
            // Check if it's an SMatrix type
            if let Some(segment) = path.segments.last() {
                let type_name = segment.ident.to_string();
                
                // Handle explicit SMatrix<T, R, C>
                if type_name == "SMatrix" {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if args.args.len() >= 3 {
                            // Extract R and C generic arguments (skip T which is first)
                            let rows = extract_const_generic(&args.args[1]);
                            let cols = extract_const_generic(&args.args[2]);
                            return FieldTypeInfo::SMatrix { rows, cols };
                        }
                    }
                    panic!("SMatrix must have type parameters SMatrix<T, R, C>");
                }
                // Handle nalgebra type aliases
                else if let Some((rows, cols)) = get_nalgebra_dimensions(&type_name) {
                    return FieldTypeInfo::SMatrix { rows, cols };
                }
                // Handle Complex<T>
                else if type_name == "Complex" {
                    return FieldTypeInfo::Complex;
                }
            }
            FieldTypeInfo::Single
        },
        _ => FieldTypeInfo::Single,
    }
}

/// Get the dimensions for known nalgebra type aliases
pub fn get_nalgebra_dimensions(type_name: &str) -> Option<(usize, usize)> {
    match type_name {
        // Vectors (column vectors)
        "Vector2" => Some((2, 1)),
        "Vector3" => Some((3, 1)),
        "Vector4" => Some((4, 1)),
        "Vector5" => Some((5, 1)),
        "Vector6" => Some((6, 1)),
        // Square matrices
        "Matrix2" => Some((2, 2)),
        "Matrix3" => Some((3, 3)),
        "Matrix4" => Some((4, 4)),
        "Matrix5" => Some((5, 5)),
        "Matrix6" => Some((6, 6)),
        // Row vectors
        "RowVector2" => Some((1, 2)),
        "RowVector3" => Some((1, 3)),
        "RowVector4" => Some((1, 4)),
        "RowVector5" => Some((1, 5)),
        "RowVector6" => Some((1, 6)),
        _ => None,
    }
}

/// Generate the appropriate nalgebra constructor for zeros
pub fn generate_nalgebra_zeros(rows: usize, cols: usize) -> proc_macro2::TokenStream {
    if cols == 1 {
        // This is a vector, use Vector constructor if available
        match rows {
            2 => quote! { nalgebra::Vector2::<T>::zeros() },
            3 => quote! { nalgebra::Vector3::<T>::zeros() },
            4 => quote! { nalgebra::Vector4::<T>::zeros() },
            5 => quote! { nalgebra::Vector5::<T>::zeros() },
            6 => quote! { nalgebra::Vector6::<T>::zeros() },
            _ => quote! { nalgebra::SMatrix::<T, #rows, #cols>::zeros() },
        }
    } else if rows == 1 {
        // This is a row vector, use RowVector constructor if available
        match cols {
            2 => quote! { nalgebra::RowVector2::<T>::zeros() },
            3 => quote! { nalgebra::RowVector3::<T>::zeros() },
            4 => quote! { nalgebra::RowVector4::<T>::zeros() },
            5 => quote! { nalgebra::RowVector5::<T>::zeros() },
            6 => quote! { nalgebra::RowVector6::<T>::zeros() },
            _ => quote! { nalgebra::SMatrix::<T, #rows, #cols>::zeros() },
        }
    } else if rows == cols {
        // This is a square matrix, use Matrix constructor if available
        match rows {
            2 => quote! { nalgebra::Matrix2::<T>::zeros() },
            3 => quote! { nalgebra::Matrix3::<T>::zeros() },
            4 => quote! { nalgebra::Matrix4::<T>::zeros() },
            5 => quote! { nalgebra::Matrix5::<T>::zeros() },
            6 => quote! { nalgebra::Matrix6::<T>::zeros() },
            _ => quote! { nalgebra::SMatrix::<T, #rows, #cols>::zeros() },
        }
    } else {
        // General case
        quote! { nalgebra::SMatrix::<T, #rows, #cols>::zeros() }
    }
}

pub fn extract_const_generic(arg: &GenericArgument) -> usize {
    match arg {
        GenericArgument::Const(expr) => {
            match expr {
                Expr::Lit(lit) => {
                    match &lit.lit {
                        syn::Lit::Int(int_lit) => {
                            int_lit.base10_parse::<usize>().unwrap_or_else(|_| {
                                panic!("Matrix dimension must be a positive integer")
                            })
                        },
                        _ => panic!("Matrix dimension must be an integer literal"),
                    }
                },
                _ => panic!("Matrix dimension must be a compile-time constant"),
            }
        },
        _ => panic!("Expected const generic argument for matrix dimension"),
    }
}
