set -e

rm build -rf
mkdir build
cd build
cmake ..
make
cd ..
mv build/resnet .
