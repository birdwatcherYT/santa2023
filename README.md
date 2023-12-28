# santa2023

- 逆の操作を記録する done
- wildcard
    - スコア評価時に考慮する
- ゴールからたどると良い場合
    - ワイルドカードあるからだめか

## 小さい問題
- 全探索
- number puzzleっぽく探索する
- 現状: スコアの小さいものから探索するようになっているため、最短とは限らない
    - ハッシュのシード変えるだけでも変わるかも？
- case 30でメモリ足りない
    - 打ち切り done
    - 訪問済みを削除
    - stateを保存せず毎回シミュレートする
    - 候補生成をランダム化して絞る
    - 悪化しすぎる候補は追加しない
    - 長さ制限をいれた探索
- 並列化

## 大きい問題
- 初期解貪欲
    - なるべく状態が近くなるように
    - sample_submissionを初期値に
- 焼き鈍す
    - 近傍
        - action削除
        - action変更
        - action交換
        - action追加
        - 破壊再構築
    - スコア
        - ターゲットとの一致具合
        - actionの長さ

系列の区間を繋ぐ操作を生成する
    ランダム性のある貪欲
    前処理で同じ状態が2度出てきたら圧縮する

IDA*
https://qiita.com/guicho271828/items/b3e885c5bde5bf7183c2
https://qiita.com/persimmon-persimmon/items/48bf1b021c349d338f0f

ゴールから展開した地点との差を評価関数にする
2ステップを1操作にする
性質を観察する
双方向IDA

並行な操作には順序をつける
3/3/3
3回転=-1回転



- 互いに独立な操作に順序をつける done
- 複数操作をコスト付きでまとめる
    ループ
- ~~mistakeを変更箇所の数で割る~~

共通のゴールを持った問題たちをまとめて解く

双方向IDA*
frontierをいくつか保持する


g(start,x)+h(x,goal)
g(goal,y)+h(y,start)
- x: frontier from start
- y: frontier from goal

g(start,x)+h(x,y)+g(y,goal)
- x: frontier from start
- y: frontier from goal


<!-- hをfrontierとのminにする -->
双方向IDA*を再帰で書く
    スタックをスカラーで持つ
g(start,x)+h(x,y)+g(y,goal)を試す
DFSの探索順序を評価値順にする
評価関数を回帰する
    pythonでNNで推論できるか確認する
    NNをC++で書く
ソルバーに投げてみる


部分的に揃える

ゴールから制約付きでランダムに動かして、パターンを記録する(これをhとしてつかう)

<!-- h := 各点を揃えるために必要な回数の和
    事前計算可能

                  A
A A A A A A A A A B B B B B B B B B C C C C C C C C C D D D D D D D D D E E E E E E E E E F F F F F F F F F -->
メモリおさえる


ルービックキューブ専用のアルゴリズム
    https://qiita.com/7y2n/items/a840e44dba77b1859352
    http://kociemba.org/cube.htm
    https://www.kaggle.com/code/wrrosa/santa-2023-kociemba-s-two-phase-algo-1-116-550


輪っか専用のアルゴリズム

<!-- ゴールからとスタートから順にランダムに伸ばしてぶつかるか確認する -->
部分問題にする
g(start,x)+h(x,y)+g(y,goal)
双方向時は上限を半分にする
    解がみつからないのでバグかも → wildcardのケース

<!-- 並列処理 -->
<!-- 近傍点がwildcard考慮してゴールかどうか -->

近傍点をランダムウォークにする？
複数操作をまとめる


キューブ以外も特化したソルバー
    wreathは作りやすそう
    回転させて2つの色が揃ったら
双方向探索で現在の解への合流、現在の解を利用
図形の固定

ヒューリスティック関数をdoubleにして、割る
ソルバーへのラベルマッピング
    → パリティが異なるのでだめ

順不同操作の置き換えによる削減

globeを解く
pqの方もdoubleにする
200からの改善はメモリがもう少し必要

1. 各地点から長さ制限しつつランダムウォークで候補生成
2. 交点を結んで改善していればマージ

双方向ビームサーチ
    top K個に絞る

大きいグラフを削減するほうが効率的
グラフごとに区間サイズを変える

