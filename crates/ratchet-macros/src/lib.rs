mod wgsl_metadata;

use proc_macro::TokenStream;
use syn::parse_macro_input;

#[proc_macro_derive(WgslMetadata, attributes(builder))]
pub fn derive_wgsl_metadata(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    wgsl_metadata::derive(input).into()
}
