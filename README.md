# santa2023

# 環境
- g++
- make
- openmp

# フォルダ構造
- data/: 問題データをAtCoder形式で保存
    - {problem_id}.txt
- output/: 解を出力
    - {problem_id}.txt
- *.cpp

# 使い方
## 解の圧縮
```bash
make
./heuristic.exe input_dir output_dir mode [problem_id]
```
- input_dir:  圧縮元フォルダ。{input_dir}/{problem_id}.txtに解が保存されている必要がある
- output_dir: 出力先フォルダ。{output_dir}/{problem_id}.txtに解を保存。フォルダは事前に作っておく必要あり。input_dir=output_dirでもよい
- mode: 4つのモード 0,1,2,3. 
    - 0: 何もしない。解のバリデーションのみ。 
    - 1: 単純な処理のみ。wildcard_finish/same_state_skip/cancel_opposite_loop/rotate_skip/summerize_rotate。
    - 2: SWAP処理のみ。find_swap。
    - 3: 重い処理。dual_greedy_improve。問題によってはメモリバク食いかつCPUフル稼働で激重。openmp必要。
    - 基本は、1実行後、2を実施すればOK。3は軽いものに絞るかベスト解のみに適用がおすすめ。
- problem_id: 圧縮したい問題ID。ない場合は全部圧縮。

例
```bash
./heuristic.exe output output 1
```

## CUBE
実装中...
