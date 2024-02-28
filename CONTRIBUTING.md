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

### Step 3: Install PyO3

Use `just` to install PyO3, ensuring Rust can interface with Python:

```sh
just install-pyo3
```

### Step 4: Configure Python Environment for PyO3

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

### Step 5: Install `pyenv` and `pyenv-virtualenv`

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

### Step 6: Set Local Python Version and Install Dependencies

Set the local Python version to 3.10.6 (or whichever version given by PyO3) and verify it:

```sh
pyenv local 3.10.6
python --version
```

Install the required Python packages specified in `requirements.txt`:

```sh
pip install -r requirements.txt
```

### Step 7: Run Tests

Finally, run the tests for the package using Cargo:

```sh
cargo test
```

### Step 8: Run WASM Tests

To run WASM tests (e.g., the whisper test) run:

```sh
just wasm-test ratchet-models chrome
```

And check the result in:

```
http://localhost:8000
```
