make
seq 0 397 | xargs -L 1 -P 4 ./heuristic.exe output output 2
