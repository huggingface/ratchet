
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

### Step 2: Install `just` Command Runner

Before installing PyO3, ensure you have `just`, a command runner that simplifies running project-specific commands, installed. If `just` is not already installed on your system, you can install it using Cargo, Rust's package manager:

```sh
cargo install just
```

This step assumes you have Rust and Cargo already installed on your system. If not, please refer to the Rust installation guide to set up Rust and Cargo.



### Step 3: Setup Python environment

#### Option 1: Using pyenv

##### Step 1: Install PyO3

Use `just` to install PyO3, ensuring Rust can interface with Python:

```sh
just install-pyo3
```

##### Step 2: Configure Python Environment for PyO3

Add the path to the Python interpreter to your shell's configuration file. This path should point to the Python version you plan to use with PyO3.

For `zsh` users, edit `~/.zshrc`:

```sh
nano ~/.zshrc
```

For `bash` users, edit `~/.bashrc`:

```sh
nano ~/.bashrc
```

Add the following line at the end of the file, replacing `<your-path>` with the actual path to your Python interpreter (e.g., `~/.pyenv/versions/3.10.6/bin/python`):

```sh
export PYO3_PYTHON=<your-path>
```

After saving the file, apply the changes:

- For `zsh`:
  ```sh
  source ~/.zshrc
  ```
- For `bash`:
  ```sh
  source ~/.bashrc
  ```

Verify the environment variable is set correctly:

```sh
echo $PYO3_PYTHON
```

##### Step 3: Install `pyenv` and `pyenv-virtualenv`

If `pyenv` is not installed, you can install it along with `pyenv-virtualenv` to manage Python versions more effectively. This step is optional but recommended.

```sh
brew install pyenv
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
```

Then, add the following lines to your shell's configuration file (`~/.zshrc` for Zsh or `~/.bashrc` for Bash):

```sh
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

And apply the changes:

- For `zsh`:
  ```sh
  source ~/.zshrc
  ```
- For `bash`:
  ```sh
  source ~/.bashrc

##### Step 4: Set Local Python Version and Install Dependencies

Set the local Python version to 3.10.6 (or whichever version given by PyO3) and verify it:

```sh
pyenv local 3.10.6
python --version
```

Install the required Python packages specified in `requirements.txt`:

```sh
pip install -r requirements.txt
```

#### Option 2: Using conda

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
##### Step 4: Test config

```
# set pyo3 to print the config
export PYO3_PRINT_CONFIG=1
cargo build
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
  lib_dir=<CONDA_PATH>/envs/ratchet/lib
  executable=<CONDA_PATH>/envs/ratchet/bin/python
  pointer_width=64
  build_flags=
  suppress_build_script_link_lines=false
```
If that looks like this, you are good to go.

Make sure you unset the `PYO3_PRINT_CONFIG`:
```
unset PYO3_PRINT_CONFIG`
```

### Step 4: Run Tests

Finally, run the tests for the package using Cargo:

```sh
cargo test
```

### Step 5: Run WASM Tests

To run WASM tests (e.g., the whisper test) run:

```sh
cd crates/ratchet-models/
wasm-pack test --chrome
```

And check the result in:

```
http://localhost:8000
```
