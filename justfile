line-count:
    cd ./crates/ratchet-core && scc -irs --exclude-file kernels
install-pyo3:
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --verbose 3.10.6
    echo "Please PYO3_PYTHON to your .bashrc or .zshrc"
wasm-all:
    wasm-pack build -s ratchet --target web -d `pwd`/target/pkg/ --release
wasm CRATE:
    wasm-pack build -s ratchet --target web -d `pwd`/target/pkg/{{CRATE}} --out-name {{CRATE}} ./crates/{{CRATE}} --release
wasm-test CRATE:
    wasm-pack test --chrome `pwd`/crates/{{CRATE}}
wasm-test-headless CRATE:
    wasm-pack test --chrome --headless `pwd`/crates/{{CRATE}}
wasm-it-headless:
    pnpm run -r test
