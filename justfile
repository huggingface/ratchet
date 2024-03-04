line-count:
    cd ./crates/ratchet-core && scc -irs --exclude-file kernels
install-pyo3:
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --verbose 3.10.6
    echo "Please PYO3_PYTHON to your .bashrc or .zshrc"
wasm-all:
    RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build -s ratchet --target web -d `pwd`/target/pkg/ --release
wasm CRATE:
    wasm-pack build -s ratchet --target web -d `pwd`/target/pkg/{{CRATE}} --out-name {{CRATE}} ./crates/{{CRATE}} --release
wasm-test CRATE BROWSER:
    wasm-pack test --{{BROWSER}} --headless `pwd`/crates/{{CRATE}}
vitest:
    pnpm run -r test
push-example EXAMPLE:
    git push {{ EXAMPLE }} `git subtree split --prefix=examples/{{EXAMPLE}}/out master`:main --force
