# 必要なライブラリをインポート
import numpy as np                     # 数値計算ライブラリ（配列操作など）
import random                         # ランダム処理（方向選択などに使用）
import matplotlib.pyplot as plt       # 結果の可視化（グラフ描画）

# ===== パラメータの設定 =====
GRID_WIDTH = 10                       # グリッドの横幅（x方向のマス数）
GRID_HEIGHT = 6                       # グリッドの縦幅（y方向のマス数）
NUM_ANTS = 20                         # 各イテレーションで動かすアリの数
NUM_ITERATIONS = 100                 # 探索を繰り返す回数
ALPHA = 1.0                           # フェロモンの重要度（大きいほどフェロモンに従う）
BETA = 2.0                            # 距離（ヒューリスティック情報）の重要度
EVAPORATION_RATE = 0.5               # フェロモンの蒸発率（古い情報の忘却）
Q = 100.0                             # フェロモン報酬の定数（短い経路に多く与える）

# スタート地点とゴール地点を指定
start = (0, 0)                        # スタートはグリッドの左上
end = (9, 5)                          # ゴールはグリッドの右下

# 各マスに対して、4方向（上下左右）にフェロモンを初期値1で設定
pheromone = np.ones((GRID_WIDTH, GRID_HEIGHT, 4))

# 4つの移動方向（上, 下, 左, 右）をベクトルで定義
directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# マスがグリッドの範囲内かを判定する関数
def is_valid(x, y):
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

# 指定したマスから移動する方向を確率的に選ぶ関数
def choose_direction(x, y, visited):
    probs = []                        # 各方向の選択確率を格納するリスト
    total = 0                         # 確率の合計
    for i, (dx, dy) in enumerate(directions):
        nx, ny = x + dx, y + dy       # 次のマスの座標を計算
        if not is_valid(nx, ny) or (nx, ny) in visited:
            probs.append(0)           # 無効なマスまたは訪問済みなら確率0
            continue
        # フェロモンと距離による選択確率の計算
        pher = pheromone[x, y, i] ** ALPHA  # 現在位置 (x, y) から方向 i にあるフェロモン量を ALPHA 乗し、影響度を調整
        cost = (1.0 / (abs(end[0] - nx) + abs(end[1] - ny) + 1e-5)) ** BETA  # ゴールとの距離の逆数を BETA 乗して重みづけ

#総移動時間の最小化ってどうするん


        prob = pher * cost  # フェロモンと距離による魅力度を掛け合わせて、その方向の選択確率の重みを計算
        probs.append(prob)  # その方向の重みをリストに追加
        total += prob  # 総和をとって、後で確率として正規化するために加算

    if total == 0:
        return None                   # どこにも進めない場合
    probs = [p / total for p in probs]  # 確率を正規化
    return random.choices(range(4), weights=probs, k=1)[0]  # 確率的に1方向を選択

# アリ1匹が経路をたどる関数
def ant_path():
    path = [start]                    # 経路にスタート地点を追加
    x, y = start
    visited = set()
    visited.add((x, y))               # 訪問済みマスに登録
    while (x, y) != end:
        direction = choose_direction(x, y, visited)
        if direction is None:
            return None               # 行き詰まりで終了
        dx, dy = directions[direction]
        x += dx
        y += dy
        if (x, y) in visited:
            return None               # ループ防止：すでに訪れたら終了
        visited.add((x, y))
        path.append((x, y))           # 経路にマスを追加
    return path                       # 最終的な経路を返す

# フェロモンマップを更新する関数
def update_pheromones(paths):
    global pheromone  # グローバル変数 pheromone を使用することを明示
    pheromone *= (1 - EVAPORATION_RATE)  # 全体的にフェロモンを蒸発させる（古い経路情報を徐々に消す）

    for path in paths:  # 各アリが見つけた経路に対して処理
        if path is None:
            continue  # 無効な経路はスキップ

        reward = Q / len(path)  # 経路が短いほどフェロモン報酬は大きくなる（効率の良い経路を強調）

        for i in range(len(path) - 1):  # 経路内の各ステップ（移動）についてフェロモンを追加
            x, y = path[i]       # 現在の位置
            nx, ny = path[i + 1] # 次の位置
            dx, dy = nx - x, ny - y  # 移動方向を計算

            for d_index, (ddx, ddy) in enumerate(directions):  # 上下左右の方向をチェック
                if (ddx, ddy) == (dx, dy):  # 実際の移動方向に一致する方向を見つけたら
                    pheromone[x, y, d_index] += reward  # その位置から（x、y）、その方向にフェロモンを追加
                    break  # 一致したらループを抜ける（無駄なチェックを避ける）

# ===== メインの繰り返しループ（ACOのコア） =====
best_path = None
for iter in range(NUM_ITERATIONS):       # 指定回数だけ試行
    paths = [ant_path() for _ in range(NUM_ANTS)]    # 各アリの経路を取得
    paths = [p for p in paths if p is not None]      # 有効な経路だけ残す
    if paths:
        best_in_iter = min(paths, key=len)           # 今回の最短経路を取得
        if best_path is None or len(best_in_iter) < len(best_path):
            best_path = best_in_iter                 # 全体のベストを更新
    update_pheromones(paths)                         # フェロモン更新

# ===== 結果の可視化 =====

# 経路をグリッドに反映（表示用）
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))          # グリッド初期化
for (x, y) in best_path:
    grid[y, x] = 1                                   # 経路マスに1を設定

# 経路を線で描画（前後のマスを線でつなぐ）
for i in range(1, len(best_path)):
    x1, y1 = best_path[i - 1]
    x2, y2 = best_path[i]
    plt.plot([x1, x2], [y1, y2], color='black', linewidth=2)

plt.title("Best Path Found by ACO")                  # グラフのタイトル
plt.show()                                            # 表示
