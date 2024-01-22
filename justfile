line-count:
    cd ./crates/ratchet-core && scc -irs --exclude-file kernels
install-pyo3:
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --verbose 3.10.6
    echo "Please PYO3_PYTHON to your .bashrc or .zshrc"
wasm CRATE:
    RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg/{{CRATE}} --out-name {{CRATE}} ./crates/{{CRATE}} --release 

