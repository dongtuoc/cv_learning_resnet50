#!/bin/bash

# 遍历当前目录下的所有子目录, 并删除目录下的 build 和 resnet 文件
for dir in */ ; do
  # 检查是否是目录
  if [ -d "$dir" ]; then
    # 进入目录
    cd "$dir"
    # 检查并删除 build 目录
    if [ -d "build" ]; then
      echo "Deleting build directory in $dir"
      rm -rf build
    fi
    # 检查并删除 resnet 可执行文件
    if [ -f "resnet" ]; then
      echo "Deleting resnet executable in $dir"
      rm -f resnet
    fi
    # 检查 codegen 文件
    if [ -d "codegen" ]; then
      echo "Deleting codegen directory in $dir"
      rm -rf codegen 
    fi
    # 返回上一级目录
    cd ..
  fi
done
echo "Cleanup completed."

