#!/usr/bin/env python3
"""
GIFファイルから画像特徴量を抽出し、PCAでクラスタリングを行うスクリプト
"""

import os
import glob
import re
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')


def load_gif_frames(gif_path, sample_frames=10):
    """GIFファイルからフレームを読み込む"""
    frames = []
    try:
        gif = Image.open(gif_path)
        n_frames = getattr(gif, 'n_frames', 1)
        
        # フレームをサンプリング
        if n_frames <= sample_frames:
            indices = range(n_frames)
        else:
            indices = np.linspace(0, n_frames - 1, sample_frames, dtype=int)
        
        for i in indices:
            gif.seek(i)
            frame = gif.convert('RGB')
            frame_np = np.array(frame)
            frames.append(frame_np)
    except Exception as e:
        print(f"Error loading {gif_path}: {e}")
    
    return frames


def extract_color_histogram(image, bins=32):
    """色ヒストグラム特徴量"""
    features = []
    for i in range(3):  # RGB
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
    return np.array(features)


def extract_hsv_histogram(image, bins=32):
    """HSVヒストグラム特徴量"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        features.extend(hist)
    return np.array(features)


def extract_hu_moments(image):
    """Huモーメント（形状特徴）"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # 対数スケールに変換（符号を保持）
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments


def extract_glcm_features(image):
    """GLCM（Gray Level Co-occurrence Matrix）テクスチャ特徴"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 256階調を16階調に削減
    gray = (gray / 16).astype(np.uint8)
    
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                        levels=16, symmetric=True, normed=True)
    
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in properties:
        values = graycoprops(glcm, prop).flatten()
        features.extend([values.mean(), values.std()])
    
    return np.array(features)


def extract_lbp_features(image, radius=3, n_points=24):
    """LBP（Local Binary Pattern）テクスチャ特徴"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # ヒストグラムを作成
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_hog_features(image):
    """HOG（Histogram of Oriented Gradients）特徴"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 画像をリサイズ（HOGは固定サイズが必要）
    resized = cv2.resize(gray, (64, 64))
    
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
    
    # 統計量に削減
    return np.array([
        features.mean(), features.std(), features.max(), features.min(),
        np.percentile(features, 25), np.percentile(features, 75)
    ])


def extract_edge_features(image):
    """エッジ特徴量"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)
    
    # Sobelエッジ
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    return np.array([
        edge_density,
        gradient_magnitude.mean(),
        gradient_magnitude.std(),
        gradient_magnitude.max()
    ])


def extract_statistical_features(image):
    """基本統計量"""
    features = []
    
    for i in range(3):  # RGB各チャンネル
        channel = image[:, :, i].flatten()
        features.extend([
            channel.mean(),
            channel.std(),
            stats.skew(channel),
            stats.kurtosis(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90)
        ])
    
    # グレースケール
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).flatten()
    features.extend([
        gray.mean(),
        gray.std(),
        stats.skew(gray),
        stats.kurtosis(gray)
    ])
    
    return np.array(features)


def extract_fft_features(image):
    """フーリエ変換特徴"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # 中心からの距離に基づいてパワースペクトルを計算
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 低周波、中周波、高周波の比率
    low_freq = magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10].sum()
    mid_freq = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30].sum() - low_freq
    high_freq = magnitude_spectrum.sum() - low_freq - mid_freq
    
    total = low_freq + mid_freq + high_freq + 1e-10
    
    return np.array([
        low_freq / total,
        mid_freq / total,
        high_freq / total,
        magnitude_spectrum.mean(),
        magnitude_spectrum.std()
    ])


def extract_spatial_distribution(image):
    """粒子の空間分布特徴"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 二値化
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # 重心
    moments = cv2.moments(binary)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = gray.shape[1] / 2, gray.shape[0] / 2
    
    # 分散
    y_indices, x_indices = np.where(binary > 0)
    if len(x_indices) > 0:
        x_var = np.var(x_indices)
        y_var = np.var(y_indices)
        spread = np.sqrt(x_var + y_var)
    else:
        x_var, y_var, spread = 0, 0, 0
    
    # 象限ごとの分布
    h, w = gray.shape
    quadrants = [
        binary[:h//2, :w//2].sum(),
        binary[:h//2, w//2:].sum(),
        binary[h//2:, :w//2].sum(),
        binary[h//2:, w//2:].sum()
    ]
    total = sum(quadrants) + 1e-10
    quadrant_ratios = [q / total for q in quadrants]
    
    return np.array([
        cx / w, cy / h,  # 正規化された重心
        x_var / (w**2), y_var / (h**2),  # 正規化された分散
        spread / np.sqrt(w**2 + h**2),  # 正規化された広がり
    ] + quadrant_ratios)


def get_feature_names():
    """特徴量の名前リストを取得"""
    names = []
    
    # Color Histogram (RGB) - 16 bins × 3 channels = 48
    for ch in ['R', 'G', 'B']:
        for i in range(16):
            names.append(f'color_hist_{ch}_bin{i}')
    
    # HSV Histogram - 16 bins × 3 channels = 48
    for ch in ['H', 'S', 'V']:
        for i in range(16):
            names.append(f'hsv_hist_{ch}_bin{i}')
    
    # Hu Moments - 7
    for i in range(7):
        names.append(f'hu_moment_{i+1}')
    
    # GLCM Features - 5 properties × 2 (mean, std) = 10
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        names.append(f'glcm_{prop}_mean')
        names.append(f'glcm_{prop}_std')
    
    # LBP Features - 26 (n_points + 2)
    for i in range(26):
        names.append(f'lbp_hist_bin{i}')
    
    # HOG Features - 6
    for stat in ['mean', 'std', 'max', 'min', 'p25', 'p75']:
        names.append(f'hog_{stat}')
    
    # Edge Features - 4
    names.append('edge_density')
    names.append('gradient_magnitude_mean')
    names.append('gradient_magnitude_std')
    names.append('gradient_magnitude_max')
    
    # Statistical Features - 6 × 3 (RGB) + 4 (gray) = 22
    for ch in ['R', 'G', 'B']:
        names.append(f'stat_{ch}_mean')
        names.append(f'stat_{ch}_std')
        names.append(f'stat_{ch}_skew')
        names.append(f'stat_{ch}_kurtosis')
        names.append(f'stat_{ch}_p10')
        names.append(f'stat_{ch}_p90')
    names.append('stat_gray_mean')
    names.append('stat_gray_std')
    names.append('stat_gray_skew')
    names.append('stat_gray_kurtosis')
    
    # FFT Features - 5
    names.append('fft_low_freq_ratio')
    names.append('fft_mid_freq_ratio')
    names.append('fft_high_freq_ratio')
    names.append('fft_magnitude_mean')
    names.append('fft_magnitude_std')
    
    # Spatial Distribution - 9
    names.append('spatial_centroid_x')
    names.append('spatial_centroid_y')
    names.append('spatial_variance_x')
    names.append('spatial_variance_y')
    names.append('spatial_spread')
    names.append('spatial_quadrant_TL')
    names.append('spatial_quadrant_TR')
    names.append('spatial_quadrant_BL')
    names.append('spatial_quadrant_BR')
    
    return names


def get_all_feature_names():
    """全特徴量の名前リスト（mean, std, temporal_mean, temporal_std を含む）"""
    base_names = get_feature_names()
    all_names = []
    
    for prefix in ['mean', 'std', 'temporal_mean', 'temporal_std']:
        for name in base_names:
            all_names.append(f'{prefix}_{name}')
    
    return all_names


def process_single_gif(gif_path):
    """
    1つのGIFファイルから特徴量を抽出する（並列処理用）
    
    Parameters:
    -----------
    gif_path : str
        GIFファイルのパス
    
    Returns:
    --------
    tuple : (gif_path, features, params) or (gif_path, None, None) if failed
    """
    try:
        frames = load_gif_frames(gif_path, sample_frames=10)
        if not frames:
            return (gif_path, None, None)
        
        features = extract_all_features(frames)
        
        # NaNやInfをチェック
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        params = parse_filename(os.path.basename(gif_path))
        return (gif_path, features, params)
    except Exception as e:
        print(f"Error processing {gif_path}: {e}")
        return (gif_path, None, None)


def save_feature_index_csv(results_dir):
    """特徴量の名前と説明を一覧にしてCSVに出力"""
    import pandas as pd
    
    base_names = get_feature_names()
    
    # 特徴量のカテゴリと説明
    feature_info = []
    
    idx = 0
    # Color Histogram
    for ch in ['R', 'G', 'B']:
        for i in range(16):
            feature_info.append({
                'index': idx,
                'name': f'color_hist_{ch}_bin{i}',
                'category': 'Color Histogram',
                'description': f'{ch}チャンネルの色ヒストグラム（bin {i}）',
                'source': 'OpenCV calcHist'
            })
            idx += 1
    
    # HSV Histogram
    for ch, ch_name in [('H', 'Hue'), ('S', 'Saturation'), ('V', 'Value')]:
        for i in range(16):
            feature_info.append({
                'index': idx,
                'name': f'hsv_hist_{ch}_bin{i}',
                'category': 'HSV Histogram',
                'description': f'{ch_name}チャンネルのHSVヒストグラム（bin {i}）',
                'source': 'OpenCV calcHist'
            })
            idx += 1
    
    # Hu Moments
    hu_descriptions = [
        '回転・スケール・反転不変',
        '回転・スケール不変',
        '回転・スケール不変',
        '回転・スケール不変',
        '回転・スケール不変（符号反転あり）',
        '回転・スケール不変（符号反転あり）',
        '反転識別'
    ]
    for i in range(7):
        feature_info.append({
            'index': idx,
            'name': f'hu_moment_{i+1}',
            'category': 'Hu Moments',
            'description': f'Huモーメント {i+1}: {hu_descriptions[i]}',
            'source': 'OpenCV HuMoments'
        })
        idx += 1
    
    # GLCM Features
    glcm_descriptions = {
        'contrast': 'コントラスト（局所的な輝度変化）',
        'dissimilarity': '非類似度（隣接ピクセルの差）',
        'homogeneity': '均質性（局所的な均一性）',
        'energy': 'エネルギー（テクスチャの均一性）',
        'correlation': '相関（線形依存性）'
    }
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        for stat in ['mean', 'std']:
            feature_info.append({
                'index': idx,
                'name': f'glcm_{prop}_{stat}',
                'category': 'GLCM (Gray Level Co-occurrence Matrix)',
                'description': f'GLCM {glcm_descriptions[prop]}の{stat}',
                'source': 'scikit-image graycoprops'
            })
            idx += 1
    
    # LBP Features
    for i in range(26):
        feature_info.append({
            'index': idx,
            'name': f'lbp_hist_bin{i}',
            'category': 'LBP (Local Binary Pattern)',
            'description': f'LBPヒストグラム（bin {i}）- テクスチャパターン',
            'source': 'scikit-image local_binary_pattern'
        })
        idx += 1
    
    # HOG Features
    hog_descriptions = {
        'mean': '平均',
        'std': '標準偏差',
        'max': '最大値',
        'min': '最小値',
        'p25': '25パーセンタイル',
        'p75': '75パーセンタイル'
    }
    for stat in ['mean', 'std', 'max', 'min', 'p25', 'p75']:
        feature_info.append({
            'index': idx,
            'name': f'hog_{stat}',
            'category': 'HOG (Histogram of Oriented Gradients)',
            'description': f'HOG特徴量の{hog_descriptions[stat]}',
            'source': 'scikit-image hog'
        })
        idx += 1
    
    # Edge Features
    edge_features = [
        ('edge_density', 'Cannyエッジ検出による密度'),
        ('gradient_magnitude_mean', 'Sobel勾配強度の平均'),
        ('gradient_magnitude_std', 'Sobel勾配強度の標準偏差'),
        ('gradient_magnitude_max', 'Sobel勾配強度の最大値')
    ]
    for name, desc in edge_features:
        feature_info.append({
            'index': idx,
            'name': name,
            'category': 'Edge Features',
            'description': desc,
            'source': 'OpenCV Canny/Sobel'
        })
        idx += 1
    
    # Statistical Features
    stat_descriptions = {
        'mean': '平均値',
        'std': '標準偏差',
        'skew': '歪度（分布の非対称性）',
        'kurtosis': '尖度（分布の尖り）',
        'p10': '10パーセンタイル',
        'p90': '90パーセンタイル'
    }
    for ch in ['R', 'G', 'B']:
        for stat in ['mean', 'std', 'skew', 'kurtosis', 'p10', 'p90']:
            feature_info.append({
                'index': idx,
                'name': f'stat_{ch}_{stat}',
                'category': 'Statistical Features',
                'description': f'{ch}チャンネルの{stat_descriptions[stat]}',
                'source': 'NumPy/SciPy stats'
            })
            idx += 1
    for stat in ['mean', 'std', 'skew', 'kurtosis']:
        feature_info.append({
            'index': idx,
            'name': f'stat_gray_{stat}',
            'category': 'Statistical Features',
            'description': f'グレースケールの{stat_descriptions[stat]}',
            'source': 'NumPy/SciPy stats'
        })
        idx += 1
    
    # FFT Features
    fft_features = [
        ('fft_low_freq_ratio', '低周波成分の比率'),
        ('fft_mid_freq_ratio', '中周波成分の比率'),
        ('fft_high_freq_ratio', '高周波成分の比率'),
        ('fft_magnitude_mean', 'パワースペクトルの平均'),
        ('fft_magnitude_std', 'パワースペクトルの標準偏差')
    ]
    for name, desc in fft_features:
        feature_info.append({
            'index': idx,
            'name': name,
            'category': 'FFT Features',
            'description': f'2Dフーリエ変換: {desc}',
            'source': 'NumPy FFT'
        })
        idx += 1
    
    # Spatial Distribution
    spatial_features = [
        ('spatial_centroid_x', '重心X座標（正規化）'),
        ('spatial_centroid_y', '重心Y座標（正規化）'),
        ('spatial_variance_x', 'X方向の分散（正規化）'),
        ('spatial_variance_y', 'Y方向の分散（正規化）'),
        ('spatial_spread', '広がり（正規化）'),
        ('spatial_quadrant_TL', '左上象限の分布比率'),
        ('spatial_quadrant_TR', '右上象限の分布比率'),
        ('spatial_quadrant_BL', '左下象限の分布比率'),
        ('spatial_quadrant_BR', '右下象限の分布比率')
    ]
    for name, desc in spatial_features:
        feature_info.append({
            'index': idx,
            'name': name,
            'category': 'Spatial Distribution',
            'description': f'空間分布: {desc}',
            'source': 'OpenCV moments'
        })
        idx += 1
    
    # DataFrameに変換して保存
    df = pd.DataFrame(feature_info)
    
    # 基本特徴量のCSV
    df.to_csv(f'{results_dir}/feature_index.csv', index=False, encoding='utf-8-sig')
    print(f"Saved {results_dir}/feature_index.csv ({len(df)} base features)")
    
    # 全特徴量（時系列統計を含む）のサマリー
    summary = []
    n_base = len(feature_info)
    for prefix, prefix_desc in [
        ('mean', 'フレーム平均'),
        ('std', 'フレーム標準偏差'),
        ('temporal_mean', '時間変化の平均'),
        ('temporal_std', '時間変化の標準偏差')
    ]:
        for i, info in enumerate(feature_info):
            summary.append({
                'index': len(summary),
                'name': f"{prefix}_{info['name']}",
                'category': info['category'],
                'temporal_stat': prefix_desc,
                'description': f"{prefix_desc}: {info['description']}",
                'source': info['source']
            })
    
    df_full = pd.DataFrame(summary)
    df_full.to_csv(f'{results_dir}/feature_index_full.csv', index=False, encoding='utf-8-sig')
    print(f"Saved {results_dir}/feature_index_full.csv ({len(df_full)} total features)")
    
    # カテゴリごとの集計
    category_summary = df.groupby('category').size().reset_index(name='count')
    category_summary['count_with_temporal'] = category_summary['count'] * 4
    category_summary.to_csv(f'{results_dir}/feature_category_summary.csv', index=False, encoding='utf-8-sig')
    print(f"Saved {results_dir}/feature_category_summary.csv")
    
    return df


def extract_all_features(frames):
    """全フレームから特徴量を抽出して平均"""
    all_features = []
    
    for frame in frames:
        frame_features = []
        
        # 各特徴量を抽出
        frame_features.extend(extract_color_histogram(frame, bins=16))
        frame_features.extend(extract_hsv_histogram(frame, bins=16))
        frame_features.extend(extract_hu_moments(frame))
        frame_features.extend(extract_glcm_features(frame))
        frame_features.extend(extract_lbp_features(frame))
        frame_features.extend(extract_hog_features(frame))
        frame_features.extend(extract_edge_features(frame))
        frame_features.extend(extract_statistical_features(frame))
        frame_features.extend(extract_fft_features(frame))
        frame_features.extend(extract_spatial_distribution(frame))
        
        all_features.append(frame_features)
    
    # フレーム間の平均と標準偏差
    all_features = np.array(all_features)
    mean_features = all_features.mean(axis=0)
    std_features = all_features.std(axis=0)
    
    # 時間変化（フレーム間の差分の統計）
    if len(frames) > 1:
        diffs = np.diff(all_features, axis=0)
        temporal_mean = np.abs(diffs).mean(axis=0)
        temporal_std = diffs.std(axis=0)
    else:
        temporal_mean = np.zeros_like(mean_features)
        temporal_std = np.zeros_like(mean_features)
    
    return np.concatenate([mean_features, std_features, temporal_mean, temporal_std])


def parse_filename(filename):
    """ファイル名からパラメータを抽出"""
    pattern = r'Swarm-c1_([\d.]+)-c2_([\d.]+)-c3_([\d.]+)-alpha_([\d.]+)\.gif'
    match = re.search(pattern, filename)
    if match:
        return {
            'c1': float(match.group(1)),
            'c2': float(match.group(2)),
            'c3': float(match.group(3)),
            'alpha': float(match.group(4))
        }
    return None


def determine_optimal_clusters(X, max_clusters=15, results_dir='results'):
    """
    Elbow methodとSilhouette methodを使用して最適なクラスタ数を決定
    
    Parameters:
    -----------
    X : ndarray
        特徴量行列 (n_samples, n_features)
    max_clusters : int
        評価する最大クラスタ数
    results_dir : str
        結果を保存するディレクトリ
    
    Returns:
    --------
    optimal_k : int
        最適なクラスタ数
    results : dict
        評価結果
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import pandas as pd
    
    print("Determining optimal number of clusters...")
    
    k_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
    
    # Elbow method: 慣性の減少率が最も急激に変化する点を探す
    # 2次微分で変曲点を探す
    inertia_diff1 = np.diff(inertias)
    inertia_diff2 = np.diff(inertia_diff1)
    elbow_k = k_range[np.argmax(inertia_diff2) + 2]  # +2 because of double diff
    
    # Silhouette method: 最大のシルエットスコアを持つk
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    # Calinski-Harabasz: 最大スコアを持つk
    calinski_k = k_range[np.argmax(calinski_scores)]
    
    # Davies-Bouldin: 最小スコアを持つk
    davies_k = k_range[np.argmin(davies_bouldin_scores)]
    
    # Davies-Bouldin法を優先（クラスタ内分散が小さく、クラスタ間距離が大きい）
    optimal_k = davies_k
    
    # 結果をプロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cluster Number Optimization', fontsize=14, fontweight='bold')
    
    # Elbow method
    ax1 = axes[0, 0]
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow: k={elbow_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score
    ax2 = axes[0, 1]
    ax2.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=silhouette_k, color='r', linestyle='--', label=f'Best: k={silhouette_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Method (higher is better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calinski-Harabasz score
    ax3 = axes[1, 0]
    ax3.plot(list(k_range), calinski_scores, 'mo-', linewidth=2, markersize=8)
    ax3.axvline(x=calinski_k, color='r', linestyle='--', label=f'Best: k={calinski_k}')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Calinski-Harabasz Index (higher is better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Davies-Bouldin score
    ax4 = axes[1, 1]
    ax4.plot(list(k_range), davies_bouldin_scores, 'co-', linewidth=2, markersize=8)
    ax4.axvline(x=davies_k, color='r', linestyle='--', label=f'Best: k={davies_k}')
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Davies-Bouldin Score')
    ax4.set_title('Davies-Bouldin Index (lower is better)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/cluster_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {results_dir}/cluster_optimization.png")
    
    # 結果をCSVに保存
    results_df = pd.DataFrame({
        'k': list(k_range),
        'inertia': inertias,
        'silhouette_score': silhouette_scores,
        'calinski_harabasz_score': calinski_scores,
        'davies_bouldin_score': davies_bouldin_scores
    })
    results_df.to_csv(f'{results_dir}/cluster_evaluation.csv', index=False)
    print(f"Saved {results_dir}/cluster_evaluation.csv")
    
    # サマリーを表示
    print(f"\n=== Cluster Number Optimization Summary ===")
    print(f"Elbow method suggests: k = {elbow_k}")
    print(f"Silhouette method suggests: k = {silhouette_k}")
    print(f"Calinski-Harabasz suggests: k = {calinski_k}")
    print(f"Davies-Bouldin suggests: k = {davies_k}")
    print(f"Final decision (Davies-Bouldin method): k = {optimal_k}")
    
    results = {
        'elbow_k': elbow_k,
        'silhouette_k': silhouette_k,
        'calinski_k': calinski_k,
        'davies_k': davies_k,
        'optimal_k': optimal_k,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': davies_bouldin_scores
    }
    
    return optimal_k, results


def remove_multicollinearity(X, threshold=0.95, results_dir='results'):
    """
    相関が高い特徴量を除去して多重共線性を除去
    
    Parameters:
    -----------
    X : ndarray
        標準化済みの特徴量行列 (n_samples, n_features)
    threshold : float
        相関係数の閾値（これ以上の相関がある場合、片方を除去）
    results_dir : str
        結果を保存するディレクトリ
    
    Returns:
    --------
    X_reduced : ndarray
        多重共線性を除去した特徴量行列
    selected_indices : list
        選択された特徴量のインデックス
    removed_info : list
        除去された特徴量の情報
    """
    import pandas as pd
    
    n_features = X.shape[1]
    feature_names = get_all_feature_names()
    
    # 相関行列を計算
    print("Computing correlation matrix...")
    corr_matrix = np.corrcoef(X.T)
    
    # NaNを0に置換
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # 除去する特徴量のインデックスを記録
    to_remove = set()
    removed_info = []
    
    # 上三角行列の要素をチェック
    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            
            corr = abs(corr_matrix[i, j])
            if corr >= threshold:
                # 相関が高い場合、j番目の特徴量を除去（後の方を除去）
                to_remove.add(j)
                removed_info.append({
                    'removed_index': j,
                    'removed_name': feature_names[j] if j < len(feature_names) else f'feature_{j}',
                    'correlated_with_index': i,
                    'correlated_with_name': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                    'correlation': corr
                })
    
    # 選択された特徴量のインデックス
    selected_indices = [i for i in range(n_features) if i not in to_remove]
    
    # 特徴量を選択
    X_reduced = X[:, selected_indices]
    
    # 除去情報をCSVに保存
    if removed_info:
        df_removed = pd.DataFrame(removed_info)
        df_removed = df_removed.sort_values('correlation', ascending=False)
        df_removed.to_csv(f'{results_dir}/removed_features_multicollinearity.csv', 
                          index=False, encoding='utf-8-sig')
        print(f"Saved {results_dir}/removed_features_multicollinearity.csv")
    
    # 選択された特徴量のリストを保存
    selected_names = [feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                      for i in selected_indices]
    df_selected = pd.DataFrame({
        'original_index': selected_indices,
        'new_index': range(len(selected_indices)),
        'name': selected_names
    })
    df_selected.to_csv(f'{results_dir}/selected_features_after_multicollinearity.csv', 
                       index=False, encoding='utf-8-sig')
    print(f"Saved {results_dir}/selected_features_after_multicollinearity.csv")
    
    # 相関行列のヒートマップを保存（除去前）
    print("Saving correlation heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 除去前の相関行列（サンプリング表示）
    sample_size = min(100, n_features)
    sample_indices = np.linspace(0, n_features - 1, sample_size, dtype=int)
    corr_sample = corr_matrix[np.ix_(sample_indices, sample_indices)]
    
    im1 = axes[0].imshow(corr_sample, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title(f'Correlation Matrix (Before)\n{n_features} features (sampled {sample_size})')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Feature Index')
    plt.colorbar(im1, ax=axes[0], label='Correlation')
    
    # 除去後の相関行列
    if len(selected_indices) > 0:
        corr_reduced = np.corrcoef(X_reduced.T)
        corr_reduced = np.nan_to_num(corr_reduced, nan=0.0)
        
        sample_size_reduced = min(100, len(selected_indices))
        sample_indices_reduced = np.linspace(0, len(selected_indices) - 1, sample_size_reduced, dtype=int)
        corr_sample_reduced = corr_reduced[np.ix_(sample_indices_reduced, sample_indices_reduced)]
        
        im2 = axes[1].imshow(corr_sample_reduced, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_title(f'Correlation Matrix (After)\n{len(selected_indices)} features (sampled {sample_size_reduced})')
        axes[1].set_xlabel('Feature Index')
        axes[1].set_ylabel('Feature Index')
        plt.colorbar(im2, ax=axes[1], label='Correlation')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/correlation_matrix_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {results_dir}/correlation_matrix_comparison.png")
    
    # 統計情報を表示
    print(f"\n=== Multicollinearity Removal Summary ===")
    print(f"Correlation threshold: {threshold}")
    print(f"Original features: {n_features}")
    print(f"Removed features: {len(to_remove)}")
    print(f"Selected features: {len(selected_indices)}")
    
    if removed_info:
        # 最も相関が高かったペアを表示
        top_corr = sorted(removed_info, key=lambda x: x['correlation'], reverse=True)[:5]
        print(f"\nTop 5 removed correlations:")
        for info in top_corr:
            print(f"  {info['removed_name']} <-> {info['correlated_with_name']}: {info['correlation']:.4f}")
    
    return X_reduced, selected_indices, removed_info


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GIF feature extraction and clustering')
    parser.add_argument('--cluster-only', action='store_true',
                        help='Skip feature extraction and use saved feature_vectors.npy')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Manually specify number of clusters (overrides Davies-Bouldin)')
    args = parser.parse_args()
    
    # resultsフォルダを作成
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # GIFファイルを検索
    gif_dir = 'gif'
    gif_files = sorted(glob.glob(os.path.join(gif_dir, '*.gif')))
    
    if not gif_files:
        print("No GIF files found in 'gif' directory")
        return
    
    print(f"Found {len(gif_files)} GIF files")
    
    # --cluster-only モードの場合、保存された特徴量を読み込む
    if args.cluster_only:
        feature_file = f'{results_dir}/feature_vectors.npy'
        metadata_file = f'{results_dir}/analysis_results.csv'
        
        if not os.path.exists(feature_file):
            print(f"Error: {feature_file} not found. Run without --cluster-only first.")
            return
        
        print(f"Loading saved features from {feature_file}...")
        X = np.load(feature_file)
        print(f"Loaded feature matrix shape: {X.shape}")
        
        # メタデータから有効なファイルリストを復元
        import pandas as pd
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            valid_files = [os.path.join(gif_dir, f) for f in df['filename'].tolist()]
            params_list = df[['c1', 'c2', 'c3', 'alpha']].to_dict('records')
        else:
            # メタデータがない場合はGIFファイルから推測
            valid_files = gif_files[:X.shape[0]]
            params_list = [parse_filename(os.path.basename(f)) for f in valid_files]
    else:
        # 特徴量を並列抽出
        features_list = []
        params_list = []
        valid_files = []
        
        # 利用可能なCPUコア数を取得（全コアの75%を使用）
        n_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        print(f"Extracting features using {n_workers} workers...")
        
        # ProcessPoolExecutorで並列処理
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 全ファイルをサブミット
            futures = {executor.submit(process_single_gif, gif_path): gif_path 
                      for gif_path in gif_files}
            
            # tqdmで進捗表示しながら結果を収集
            for future in tqdm(as_completed(futures), total=len(gif_files)):
                gif_path, features, params = future.result()
                if features is not None:
                    features_list.append(features)
                    params_list.append(params)
                    valid_files.append(gif_path)
        
        if not features_list:
            print("No valid features extracted")
            return
        
        # 結果をファイル名でソート（順序を保持するため）
        sorted_data = sorted(zip(valid_files, features_list, params_list), 
                            key=lambda x: x[0])
        valid_files = [x[0] for x in sorted_data]
        features_list = [x[1] for x in sorted_data]
        params_list = [x[2] for x in sorted_data]
        
        # 特徴量行列
        X = np.array(features_list)
        print(f"Feature matrix shape: {X.shape}")
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # NaNを0に置換（標準化後に発生する可能性）
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 多重共線性の除去（相関が高い特徴量を除去）
    X_reduced, selected_indices, removed_info = remove_multicollinearity(
        X_scaled, threshold=0.95, results_dir=results_dir
    )
    print(f"After removing multicollinearity: {X_reduced.shape[1]} features (removed {X_scaled.shape[1] - X_reduced.shape[1]})")
    
    # PCA
    n_components = min(50, X_reduced.shape[0], X_reduced.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_reduced)
    
    print(f"PCA explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative variance (10 components): {pca.explained_variance_ratio_[:10].sum():.3f}")
    
    # クラスタ数を決定
    if args.n_clusters is not None:
        # 手動で指定されたクラスタ数を使用
        n_clusters = args.n_clusters
        print(f"Using manually specified number of clusters: {n_clusters}")
        cluster_eval_results = None
    else:
        # 最適なクラスタ数を自動決定（Davies-Bouldin法を優先）
        n_clusters, cluster_eval_results = determine_optimal_clusters(
            X_pca[:, :10], max_clusters=15, results_dir=results_dir
        )
        print(f"Optimal number of clusters (Davies-Bouldin): {n_clusters}")
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca[:, :10])  # 上位10成分を使用
    
    # パラメータ配列を作成
    c1_values = np.array([p['c1'] for p in params_list])
    c2_values = np.array([p['c2'] for p in params_list])
    c3_values = np.array([p['c3'] for p in params_list])
    alpha_values = np.array([p['alpha'] for p in params_list])
    
    # 可視化
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PCA散布図（PC1 vs PC2）- クラスタで色分け
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA (PC1 vs PC2) - Clusters')
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # 2. PCA散布図（PC1 vs PC2）- c1で色分け
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=c1_values, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA (PC1 vs PC2) - c1')
    plt.colorbar(scatter2, ax=ax2, label='c1')
    
    # 3. PCA散布図（PC1 vs PC2）- c3で色分け
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=c3_values, cmap='plasma', alpha=0.7)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('PCA (PC1 vs PC2) - c3')
    plt.colorbar(scatter3, ax=ax3, label='c3')
    
    # 4. 累積寄与率
    ax4 = fig.add_subplot(2, 3, 4)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    ax4.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
    ax4.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax4.set_xlabel('Number of Components')
    ax4.set_ylabel('Cumulative Explained Variance')
    ax4.set_title('PCA Cumulative Variance')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 3D PCA散布図
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter5 = ax5.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                           c=clusters, cmap='tab10', alpha=0.7)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_zlabel('PC3')
    ax5.set_title('3D PCA - Clusters')
    
    # 6. クラスタごとのパラメータ分布
    ax6 = fig.add_subplot(2, 3, 6)
    cluster_params = []
    for i in range(n_clusters):
        mask = clusters == i
        cluster_params.append({
            'c1_mean': c1_values[mask].mean(),
            'c3_mean': c3_values[mask].mean(),
            'alpha_mean': alpha_values[mask].mean(),
            'count': mask.sum()
        })
    
    x = np.arange(n_clusters)
    width = 0.25
    ax6.bar(x - width, [cp['c1_mean'] for cp in cluster_params], width, label='c1 mean')
    ax6.bar(x, [cp['c3_mean'] for cp in cluster_params], width, label='c3 mean')
    ax6.bar(x + width, [cp['alpha_mean'] for cp in cluster_params], width, label='alpha mean')
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Parameter Mean')
    ax6.set_title('Cluster Parameter Distribution')
    ax6.set_xticks(x)
    ax6.legend()
    
    # 特徴量一覧をCSVに出力
    save_feature_index_csv(results_dir)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/analysis_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {results_dir}/analysis_result.png")
    
    # 結果をCSVに保存
    import pandas as pd
    
    results_df = pd.DataFrame({
        'filename': [os.path.basename(f) for f in valid_files],
        'c1': c1_values,
        'c2': c2_values,
        'c3': c3_values,
        'alpha': alpha_values,
        'cluster': clusters,
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2]
    })
    results_df.to_csv(f'{results_dir}/analysis_results.csv', index=False)
    print(f"Saved {results_dir}/analysis_results.csv")
    
    # 潜在ベクトル（PCA後）を保存
    np.save(f'{results_dir}/latent_vectors.npy', X_pca)
    np.save(f'{results_dir}/feature_vectors.npy', X)
    print(f"Saved {results_dir}/latent_vectors.npy and {results_dir}/feature_vectors.npy")
    
    # クラスタ情報を表示
    print("\n=== Cluster Summary ===")
    for i in range(n_clusters):
        mask = clusters == i
        print(f"\nCluster {i}: {mask.sum()} samples")
        print(f"  c1: {c1_values[mask].mean():.2f} ± {c1_values[mask].std():.2f}")
        print(f"  c2: {c2_values[mask].mean():.2f} ± {c2_values[mask].std():.2f}")
        print(f"  c3: {c3_values[mask].mean():.2f} ± {c3_values[mask].std():.2f}")
        print(f"  alpha: {alpha_values[mask].mean():.2f} ± {alpha_values[mask].std():.2f}")
    
    # 各クラスタの代表的なムービーの最後のフレームをグリッド表示
    print("\nGenerating cluster representative frames grid...")
    create_cluster_grid(valid_files, clusters, n_clusters, results_dir)


def get_last_frame(gif_path):
    """GIFファイルから最後のフレームを取得"""
    try:
        gif = Image.open(gif_path)
        n_frames = getattr(gif, 'n_frames', 1)
        gif.seek(n_frames - 1)  # 最後のフレームに移動
        frame = gif.convert('RGB')
        return np.array(frame)
    except Exception as e:
        print(f"Error loading last frame from {gif_path}: {e}")
        return None


def create_cluster_grid(valid_files, clusters, n_clusters, results_dir, samples_per_cluster=5):
    """各クラスタの代表的なムービーの最後のフレームをグリッド表示"""
    
    # クラスタごとにファイルを分類
    cluster_files = {i: [] for i in range(n_clusters)}
    for file_path, cluster_id in zip(valid_files, clusters):
        cluster_files[cluster_id].append(file_path)
    
    # グリッドのサイズを計算
    n_rows = n_clusters
    n_cols = samples_per_cluster
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    fig.suptitle('Representative Last Frames by Cluster', fontsize=16, fontweight='bold')
    
    for cluster_id in range(n_clusters):
        files = cluster_files[cluster_id]
        
        # クラスタ内からサンプルを選択（ランダムまたは先頭から）
        if len(files) >= samples_per_cluster:
            # PCA空間でクラスタ中心に近いサンプルを選択
            np.random.seed(42 + cluster_id)
            selected_indices = np.random.choice(len(files), samples_per_cluster, replace=False)
            selected_files = [files[i] for i in selected_indices]
        else:
            # サンプル数が足りない場合は全て使用
            selected_files = files + [None] * (samples_per_cluster - len(files))
        
        for col, file_path in enumerate(selected_files):
            ax = axes[cluster_id, col] if n_clusters > 1 else axes[col]
            
            if file_path is not None:
                frame = get_last_frame(file_path)
                if frame is not None:
                    ax.imshow(frame)
                    # ファイル名からパラメータを抽出して表示
                    basename = os.path.basename(file_path)
                    # 短いラベルを作成
                    params = parse_filename(basename)
                    if params:
                        label = f"c1={params['c1']:.1f}\nα={params['alpha']:.1f}"
                        ax.set_xlabel(label, fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('lightgray')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 行の最初のセルにクラスタラベルを追加
            if col == 0:
                ax.set_ylabel(f'Cluster {cluster_id}\n({len(files)} samples)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/cluster_frames_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {results_dir}/cluster_frames_grid.png")


if __name__ == '__main__':
    main()

