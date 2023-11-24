# Release Version 
# -Ofast is fater than -O3
g++ -mavx -Ofast main.cc -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -ldl -o resnet 

# Debug Version 
# g++ -mavx -O0 -g main.cc -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -o resnet

