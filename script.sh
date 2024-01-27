make
seq 0 397 | xargs -L 1 -P 4 -I{} ./heuristic.exe output output 2 {} 50 1000
# seq 0 280 | xargs -L 1 -P 4 -I{} ./heuristic.exe output output 2 {} 50 1000
# seq 281 283 | xargs -L 1 -P 4 ./heuristic.exe output output 2
# seq 284 397 | xargs -L 1 -P 4 ./heuristic.exe output output 2
