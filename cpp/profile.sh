g++ -std=c++17 -O3 -ffast-math -flto -g -o prof prof.cpp
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./prof
kcachegrind
