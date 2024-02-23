#include <sstream>
#include <fstream>
#include <string>
#include <iostream>

static inline void write(const std::string& filename, const std::ostringstream& content) {
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  if (file.is_open()) {
    file << content.str();
    file.close();
    std::cout << "GenCode to " << filename << std::endl;
  } else {
    std::cout << "Fail to open" << filename << std::endl;
  }
}

int main() {
  std::ostringstream os;

  int v1[5] = {1, 2, 4, 5, 6};
  int v2[5] = {11, 12, 14, 15, 16};
  int res[5] = {0};

  os << "int main() {\n";
  os << "  int v1[5] = {1, 2, 4, 5, 6};\n";
  os << "  int v2[5] = {11, 12, 14, 15, 16};\n";
  os << "  int res[5] = {0};\n";

  for (int i = 0; i < 5; i++) {
    if (v1[i] % 2 == 0) {
      res[i] = v1[i] + v2[i];
      os << "  res[" << i << "] = v1[" << i << "] + v2[" << i << "];\n";
    }
  }
  os << "  return 0;\n";
  os << "};\n";

  write("code_gen_main.cc", os);

  // check result
  // for (int i = 0; i < 5; i++) {
  //   printf("%d ", res[i]);
  // }
  // printf("\n");
  return 0;
}
