line-count:
    cd ./crates/ratchet-core && scc -irs
install-pyo3:
    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --verbose 3.10.6
    pyenv local 3.10.6
    echo $(python --version)
wasm CRATE:
    node_modules/.bin/wasm-pack build -s ratchet --target web -d `pwd`/target/pkg/{{CRATE}} --out-name {{CRATE}} ./crates/{{CRATE}} --release
wasm-dbg CRATE:
    node_modules/.bin/wasm-pack build -s ratchet --target web -d `pwd`/target/pkg/{{CRATE}} --out-name {{CRATE}} ./crates/{{CRATE}} --dev
wasm-test CRATE BROWSER:
    cp ./config/webdriver-macos.json ./crates/{{CRATE}}/webdriver.json
    node_modules/.bin/wasm-pack test --{{BROWSER}} --headless `pwd`/crates/{{CRATE}}
wasm-publish-pr CRATE: # Publish a new version of a crate using pkg.pr.new
    node_modules/.bin/pkg-pr-new publish --pnpm ./target/pkg/{{CRATE}}
push-example EXAMPLE:
    git push {{ EXAMPLE }} `git subtree split --prefix=examples/{{EXAMPLE}}/out master`:main --force
export-libtorch: 
    export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)') 
    export DYLD_LIBRARY_PATH=${LIBTORCH}/lib
