#pragma once

#define CODE_GEN (1)
#include <iostream>
#include <string>

static inline void write(const std::string& filename, const std::ostringstream& content) {
  std::ofstream file(filename, std::ios::app);
  if (file.is_open()) {
    file << content.str();
    file.close();
    std::cout << "GenCode to " << filename << std::endl;
  } else {
    std::cout << "Fail to open" << filename << std::endl;
  }
}
