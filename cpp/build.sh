rm a.out

# -Ofast is fater than -O3
g++ -mavx -Ofast main.cc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

