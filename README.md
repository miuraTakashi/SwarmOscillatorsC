# Swarm Oscillators Simulation

スワーム振動子のシミュレーションと画像特徴量解析を行うプロジェクトです。粒子間の相互作用により集団的なパターンが形成される様子をシミュレートし、機械学習によるパターン分類を行います。

## 概要

このシミュレーションは、位相結合振動子と空間的な運動を組み合わせたモデルです。各粒子は：
- 位相 ψ を持ち、他の粒子との相互作用で変化
- 2次元平面上を移動し、他の粒子との距離に応じて引力/斥力を受ける

## モデル方程式

```
dψᵢ/dt = Σⱼ exp(-rᵢⱼ) sin(ψⱼ - ψᵢ + α·rᵢⱼ - c₁)

drᵢ/dt = c₃ Σⱼ (r̂ᵢⱼ) exp(-rᵢⱼ) sin(ψⱼ - ψᵢ + α·rᵢⱼ - c₂)
```

### パラメータ
| パラメータ | 説明 |
|-----------|------|
| `c1` | 位相結合の位相シフト |
| `c2` | 空間運動の位相シフト |
| `c3` | 空間運動の強度 |
| `alpha` | 距離依存の位相シフト係数 |

## プロジェクト構成

```
SwarmOscillatorsC/
├── SwarmOscillators.c     # メインのシミュレーションコード (C言語)
├── Makefile               # ビルド設定
├── analyze_gifs.py        # 画像特徴量解析・クラスタリング (Python)
├── requirements.txt       # Python依存関係
├── README.md              # このファイル
├── .gitignore             # Git除外設定
│
├── gif/                   # 生成されたGIFファイル (gitignore)
├── results/               # 解析結果
│
└── obsolete/              # 使用されていないファイル (gitignore)
    ├── MT.h               # Mersenne Twister 乱数生成器
    ├── fftw3.h            # FFTW3ライブラリヘッダ
    ├── gifenc.h           # GIFエンコーダ（ffmpegに移行）
    └── SwarmVisualization.nb  # Mathematica可視化ノートブック
    ├── analysis_result.png
    ├── analysis_results.csv
    ├── cluster_frames_grid.png
    ├── cluster_optimization.png
    ├── cluster_evaluation.csv
    ├── correlation_matrix_comparison.png
    ├── feature_index.csv
    ├── feature_index_full.csv
    ├── feature_category_summary.csv
    ├── removed_features_multicollinearity.csv
    ├── selected_features_after_multicollinearity.csv
    ├── latent_vectors.npy
    └── feature_vectors.npy
```

## クイックスタート

### 1. シミュレーション

```bash
# ビルド（OpenMP並列化有効）
make parallel

# 実行
./swarm
```

### 2. 画像解析

```bash
# Python環境セットアップ
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 解析実行（並列処理で高速）
python analyze_gifs.py

# 保存済み特徴量を使ってクラスタリングのみ（高速）
python analyze_gifs.py --cluster-only

# クラスタ数を手動指定
python analyze_gifs.py --cluster-only --n-clusters 12
```

---

## シミュレーション詳細

### 必要条件
- GCC (OpenMP対応、macOSではHomebrew経由でgcc-15推奨)
- FFTW3ライブラリ
- ffmpeg（GIF生成用）
- Make

### ビルドオプション

```bash
make              # 最適化ビルド（デフォルト）
make optimized    # 最適化ビルド（-O3 -march=native -ffast-math）
make parallel     # OpenMP並列化有効
make debug        # デバッグビルド
make clean        # クリーンアップ
make help         # ヘルプ表示
```

### シミュレーション設定

`SwarmOscillators.c` 内の定数で設定を変更：

```c
#define N_PARTICLES 50      // 粒子数
#define L 25                // シミュレーション領域サイズ
#define DT 0.05             // 時間刻み
#define LOOP 20000          // シミュレーションステップ数

#define IMG_WIDTH 128       // GIF画像幅
#define IMG_HEIGHT 128      // GIF画像高さ
#define FRAME_SKIP 50       // フレーム保存間隔
#define POINT_SIZE 2        // 粒子描画サイズ

#define PARAM_MIN 0.00      // パラメータ最小値
#define PARAM_MAX 4.50      // パラメータ最大値
#define PARAM_STEP 0.50     // パラメータ刻み幅（10⁴=10,000通り）
```

### 出力
- `gif/Swarm-c1_X.XX-c2_X.XX-c3_X.XX-alpha_X.XX.gif`

---

## 画像解析詳細

### 抽出される特徴量（740次元 → 544次元）

| カテゴリ | 特徴量 | 次元数 |
|----------|--------|--------|
| **Color Histogram** | RGB各チャンネル16bin | 48 |
| **HSV Histogram** | H(16), S(16), V(16) | 48 |
| **Hu Moments** | 7つの不変モーメント | 7 |
| **GLCM** | contrast, dissimilarity, homogeneity, energy, correlation, ASM | 6 |
| **LBP** | Local Binary Pattern (26 bins) | 26 |
| **HOG** | Histogram of Oriented Gradients (36 bins) | 36 |
| **Edge Features** | Canny edge statistics | 5 |
| **Statistical** | mean, std, skewness, kurtosis (per channel) | 12 |
| **FFT** | 周波数帯域比率、マグニチュード統計 | 5 |
| **Spatial Distribution** | 重心、分散、象限分布 | 9 |

各特徴量に対して：
- `mean_*`: フレーム平均
- `std_*`: フレーム標準偏差
- `temporal_mean_*`: 時間変化の平均
- `temporal_std_*`: 時間変化の標準偏差

合計: 185 × 4 = **740次元** → 多重共線性除去後 **544次元**

### 解析パイプライン

```
1. 特徴量抽出（並列処理: 7ワーカー）
   └── 625 GIFs → 740次元特徴ベクトル

2. 前処理
   ├── StandardScaler標準化
   └── 多重共線性除去（相関 ≥ 0.95 → 544次元）

3. 次元削減
   └── PCA（50成分、上位10成分をクラスタリングに使用）

4. クラスタ数最適化
   ├── Elbow method
   ├── Silhouette method
   ├── Calinski-Harabasz Index
   └── Davies-Bouldin Index（優先）

5. K-meansクラスタリング
```

### コマンドラインオプション

```bash
python analyze_gifs.py [OPTIONS]

Options:
  --cluster-only    保存済みfeature_vectors.npyを使用（特徴量抽出をスキップ）
  --n-clusters N    クラスタ数を手動指定（Davies-Bouldinの代わりに）
  --no-parallel     並列処理を無効化（フリーズする場合に使用）
```

### 出力ファイル

| ファイル | 説明 |
|----------|------|
| `analysis_result.png` | PCA散布図、累積寄与率、クラスタ分布 |
| `analysis_results.csv` | ファイル名、パラメータ、クラスタID、PC1-3 |
| `cluster_frames_grid.png` | クラスタごとの代表フレーム |
| `cluster_optimization.png` | クラスタ数最適化の4手法比較 |
| `cluster_evaluation.csv` | k=2〜15の各指標値 |
| `correlation_matrix_comparison.png` | 多重共線性除去前後の相関行列 |
| `feature_index.csv` | 基本特徴量のインデックス（185件） |
| `feature_index_full.csv` | 全特徴量のインデックス（740件） |
| `removed_features_multicollinearity.csv` | 除去された特徴量（196件） |
| `selected_features_after_multicollinearity.csv` | 選択された特徴量（544件） |
| `feature_vectors.npy` | 生の特徴ベクトル（625×740） |
| `latent_vectors.npy` | PCA後の潜在ベクトル（625×50） |

---

## 最適化

### シミュレーション（C言語）
1. **冗長計算の削減**: 距離計算、exp、sinなどを1回だけ計算
2. **コンパイラ最適化**: `-O3 -march=native -ffast-math`
3. **OpenMP並列化**: パラメータスキャンを4重ループで並列実行
4. **ffmpegによるGIF生成**: PPM→GIF変換で高品質・小ファイルサイズ

### 画像解析（Python）
1. **マルチプロセス並列化**: `ProcessPoolExecutor`（CPUコアの75%使用）
2. **多重共線性除去**: 相関 ≥ 0.95 の冗長特徴量を削除
3. **キャッシュ機能**: `--cluster-only`で特徴量計算をスキップ

**速度比較**:
| 処理 | シーケンシャル | 並列 | 高速化 |
|------|---------------|------|--------|
| 特徴量抽出 | ~66秒 | ~12秒 | **5.5倍** |

---

## ライセンス

- MT.h: Mersenne Twister by Makoto Matsumoto and Takuji Nishimura (BSD License)

## 参考文献

- Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
- O'Keeffe, K. P., Hong, H., & Strogatz, S. H. (2017). Oscillators that sync and swarm.
