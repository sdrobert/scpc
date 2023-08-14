# Instructions for installing KenLM
# This is a prototypical install. Use comments to guide your version

# activate your scpc environment, e.g.
conda activate scpc

# install cmake and a C++ compiler, e.g.
conda install -c conda-forge cmake gxx_linux-64

# clone repo
tmp="$(mktemp -d)"
git clone "https://github.com/kpu/kenlm.git" "$tmp" --depth 1

# build library and install into conda/python environment
cmake -B "$tmp/build" "$tmp" -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
cmake --build "$tmp/build" --parallel 16 --target install

# delete repo
rm -rf "$tmp"
