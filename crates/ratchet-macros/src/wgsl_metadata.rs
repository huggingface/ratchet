use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse2, DeriveInput};

pub fn derive(input: TokenStream) -> TokenStream {
    let _input = parse2::<DeriveInput>(input).unwrap();
    let struct_name = _input.ident;

    let syn::Data::Struct(syn::DataStruct { fields, .. }) = _input.data else {
        unimplemented!("Only structs are supported");
    };

    let transformed_fields = fields.iter().map(|field| {
        let Some(ident) = &field.ident else {
            unimplemented!("tuple structs");
        };

        let ty = &field.ty;

        match ty {
            syn::Type::Path(p) => {
                let path = &p.path;
                let t = path.segments.last().unwrap().ident.to_string();

                match t.as_str() {
                    "UVec4" => {
                        quote!(#ident: vec4<u32>)
                    }
                    "IVec4" => {
                        quote!(#ident: vec4<i32>)
                    }
                    "UVec3" => {
                        quote!(#ident: vec3<u32>)
                    }
                    "IVec3" => {
                        quote!(#ident: vec3<i32>)
                    }
                    _ => quote!(#ident: #ty),
                }
            }
            _ => todo!(),
        }
    });

    let expanded = quote! (
        impl crate::OpMetadata for #struct_name {
            fn render() -> crate::WgslFragment {
                let mut fragment = crate::WgslFragment::new(512);
                fragment.write("struct Meta {\n");
                #(
                    fragment.write("    ");
                    fragment.write(stringify!(#transformed_fields));
                    fragment.write(",\n");
                )*
                fragment.write("}\n");
                fragment
            }
        }
    );

    expanded
}
