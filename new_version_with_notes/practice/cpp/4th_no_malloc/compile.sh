set -e

rm build -rf
mkdir build
cd build
cmake ..
make -j16
cd ..
mv build/resnet .
