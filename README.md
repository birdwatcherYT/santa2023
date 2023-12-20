# santa2023

- 逆の操作を記録する done
- wildcard
    - スコア評価時に考慮する

## 小さい問題
- 全探索
- number puzzleっぽく探索する

## 大きい問題
- 初期解貪欲
    - なるべく状態が近くなるように
- 焼き鈍す
    - 近傍
        - action削除
        - action変更
        - action交換
        - action追加
    - スコア
        - ターゲットとの一致具合
        - actionの長さ
