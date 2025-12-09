# Makefile for SwarmOscillators

# コンパイラ設定（OpenMP対応のgccを使用）
CC = /opt/homebrew/bin/gcc-15
TARGET = swarm

# ソースファイル
SRC = SwarmOscillators.c

# ライブラリ
LIBS = -lm -lfftw3

# 最適化フラグ（OpenMP有効）
CFLAGS_OPT = -O3 -march=native -ffast-math -fopenmp

# デバッグフラグ
CFLAGS_DEBUG = -g -Wall -Wextra

# OpenMP並列化フラグ（オプション）
CFLAGS_OMP = -fopenmp

# デフォルトターゲット（最適化ビルド）
all: optimized

# 最適化ビルド
optimized: $(SRC)
	$(CC) $(CFLAGS_OPT) $(SRC) -o $(TARGET) $(LIBS)
	@echo "Built with optimization flags: $(CFLAGS_OPT)"

# OpenMP並列化 + 最適化ビルド
parallel: $(SRC)
	$(CC) $(CFLAGS_OPT) $(CFLAGS_OMP) $(SRC) -o $(TARGET) $(LIBS)
	@echo "Built with optimization + OpenMP: $(CFLAGS_OPT) $(CFLAGS_OMP)"

# デバッグビルド
debug: $(SRC)
	$(CC) $(CFLAGS_DEBUG) $(SRC) -o $(TARGET)_debug $(LIBS)
	@echo "Built debug version: $(TARGET)_debug"

# クリーンアップ
clean:
	rm -f $(TARGET) $(TARGET)_debug a.out

# 実行
run: optimized
	./$(TARGET)

# ヘルプ
help:
	@echo "利用可能なターゲット:"
	@echo "  make           - 最適化ビルド（デフォルト）"
	@echo "  make optimized - 最適化ビルド"
	@echo "  make parallel  - 最適化 + OpenMP並列化ビルド"
	@echo "  make debug     - デバッグビルド"
	@echo "  make run       - ビルド後に実行"
	@echo "  make clean     - 生成ファイルを削除"

.PHONY: all optimized parallel debug clean run help

