#!/bin/sh

if [ "$(uname)" = "Darwin" ]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [ "$(uname)" != "Linux" ]; then
  echo "Unsupported operating system."
  exit 1
fi

if ! [ -d "venv" ]; then
  echo "Create venv..."
  # Check if Python 3.8 is installed
  if command -v python3.8 >/dev/null 2>&1 || command -v pyenv > /dev/null 2>&1 && pyenv versions --bare | grep -q "3.8"; then
    test
  else
    echo "Python 3 not found. Attempting to install 3.8..."
    if ! command -v pyenv; then
      if [ "$(uname)" = 'Darwin' ]; then
        brew install pyenv
      elif [ "$(uname)" = 'Linux' ]; then
        if command -v apt; then
          sudo apt update
          sudo apt install -y pyenv
        elif command -v pacman; then
          sudo pacman -Sy --no-confirm pyenv
        elif command -v dnf; then
          sudo dnf install pyenv
        else
          echo "Manually install pyenv!"
          exit 1
        fi
      fi
    fi
    pyenv install 3.8
  fi
  path=$(realpath $HOME/.pyenv/versions/3.8*)
  export PATH=$path/bin:$PATH
  python3.8 -m venv venv
fi
. venv/bin/activate
requirements_file="requirements.txt"
# Check if required packages are installed and install them if not
if [ -f "${requirements_file}" ]; then
  installed_packages=$(python3.8 -m pip freeze)
  while IFS= read -r package; do
    expr "${package}" : "^#.*" > /dev/null && continue
    package_name=$(echo "${package}" | sed 's/[<>=!].*//')
    if ! echo "${installed_packages}" | grep -q "${package_name}"; then
      echo "${package_name} not found. Attempting to install..."
      python3.8 -m pip install --upgrade "${package}"
    fi
  done < "${requirements_file}"
else
  echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
  exit 1
fi

# Download models
chmod +x tools/dlmodels.sh
./tools/dlmodels.sh

if [ $? -ne 0 ]; then
  exit 1
fi
