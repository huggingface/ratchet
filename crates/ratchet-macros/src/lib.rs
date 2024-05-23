mod wgsl_metadata;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Derives the `OpMetadata` trait implementation for a struct.
///
/// Generates a `.render()` method that converts a Rust struct into a WGSL struct.
#[proc_macro_derive(WgslMetadata, attributes(builder))]
pub fn derive_wgsl_metadata(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    wgsl_metadata::derive(input).into()
}
