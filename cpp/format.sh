#!/bin/bash

echo $1
doFormatDir() {
  if [ -d $1 ];then
    pushd $1
      for x in *
        do
          doFormatDir $x
        done
    popd
  elif [ -f $1 ];then
    if [ ${1##*.} == 'cpp' ] || [ ${1##*.} == 'cc' ] || [ ${1##*.} == 'h' ] || [ ${1##*.} == 'c' ];then
      echo -e "\033[32mclang-format -i $1\033[0m"
      clang-format -i $1
    else
      echo -e "\033[32mUnknow file type($1), Skip...\033[0m"
    fi
  else
    echo -e "\033[32mUnknow file type($1), Skip...\033[0m"
  fi
}

until [[ $# -eq 0 ]]
do
doFormatDir $1
shift
done
