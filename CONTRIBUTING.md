# Running Tests for the Ratchet Rust Package

This guide outlines the steps necessary to set up and run tests for the Ratchet Rust package. Please follow these steps carefully to ensure a smooth testing process.

## Prerequisites

- Git
- Rust and Cargo

## Setup Instructions

### Step 1: Clone the Repository

First, clone the Ratchet repository from GitHub and navigate into the project directory:

```sh
git clone https://github.com/FL33TW00D/ratchet.git
cd ratchet/
```

### Option 1: Using pyenv

#### Step 1: Install `pyenv`

First, make sure to install [pyenv](https://github.com/pyenv/pyenv#getting-pyenv). `pyenv` lets you manage multiple versions of Python. Please make sure you follow the install guide and source the correct environment variables.

#### Step 2: Install `just` Command Runner

Before installing PyO3, ensure you have `just`, a command runner that simplifies running project-specific commands, installed. If `just` is not already installed on your system, you can install it using Cargo, Rust's package manager:

```sh
cargo install just
```

This step assumes you have Rust and Cargo already installed on your system. If not, please refer to the Rust installation guide to set up Rust and Cargo.

#### Step 3: Install python 3.10.6

Use `just` to install `python3.10.6` and enable it as the local python version for the project.

> **NOTE** : `PyO3`\*\* needs Python to be built with `enable-shared` flag.

```sh
just install-pyo3
```

#### Step 4: Create virtual environment (Optional)

This step is optional but _highly_ recommended. You should create and source a virtual environment using your favorite tool (`uv`, `venv`, `virtualenv`...). We'll use the built-in `venv` module:

```sh
python -m venv venv
source venv/bin/activate
```

#### Step 5: Install python dependencies

Install the Python dependencies recursively:

```sh
python -m pip install -r requirements.txt
```

##### Step 6: Configure Python Environment for PyO3

PyO3 uses a build script to determine the Python version and set the correct linker arguments. To override the Python interpreter to the virtual environment, run the following:

```sh
export PYO3_PYTHON=$(which python)
echo $PYO3_PYTHON
```

### Option 2: Using conda

##### Step 1: Create a new conda environment

```
conda create -n ratchet python=3.10
```

##### Step 2: Install dependencies

```
pip install -r requirements.txt
```

##### Step 3: Configure Cargo

Edit `<ROOT>/.cargo/config.toml` to add the linker config:

```
# .cargo/config.toml
[build]
rustflags = [
    "--cfg=web_sys_unstable_apis",
    # Add these two lines and replace PATH_TO_CONDA with your conda directory:
    "-C",
    "link-args=-Wl,-rpath,<PATH_TO_CONDA>/envs/ratchet/lib/",
]
```

### Step 2: Test config

We'll first verify that your pyo3 config is correctly setup:

```
PYO3_PRINT_CONFIG=1 cargo build
```

Building the project will throw an error(!) and print the config:

```
(exit status: 101)
  --- stdout
  cargo:rerun-if-env-changed=PYO3_PRINT_CONFIG

  -- PYO3_PRINT_CONFIG=1 is set, printing configuration and halting compile --
  implementation=CPython
  version=3.10
  shared=true
  abi3=false
  lib_name=python3.10
  lib_dir=<LOCAL PYTHON LIB>
  executable=<LOCAL PYTHON EXECUTABLE PATH>
  pointer_width=64
  build_flags=
  suppress_build_script_link_lines=false
```

If that looks like this, you are good to go 🎉

### Step 3: Run Tests

Finally, run the tests for the package using Cargo:

```sh
cargo test
```

To run the `PyO3` tests, add the `pyo3` flag:

```sh
cargo test --features pyo3
```

### Step 5: Run WASM Tests

To run WASM tests (e.g., the whisper test) run:

```sh
just wasm-test ratchet-models chrome
```

And check the result in:

```
http://localhost:8000
```
