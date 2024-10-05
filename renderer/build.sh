# This renderer is borrowed from StereoPIFu: https://github.com/CrisHY1995/StereoPIFu_Code
mkdir build
cd build
cmake .. -Dpybind11_DIR=$pybind_path
make
