#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <string.h>
#include <omp.h>

#define N_PARTICLES 50
#define L 25
#define DT 0.05
#define LOOP 20000

// 画像設定
#define IMG_WIDTH 128
#define IMG_HEIGHT 128
#define FRAME_SKIP 50    // 50ステップごとにフレームを追加
#define POINT_SIZE 2     // 粒子の描画サイズ

// パラメータスキャン設定
#define PARAM_MIN 0.00
#define PARAM_MAX 4.50
#define PARAM_STEP 0.50
#define N_PARAMS 10  // (4.50 - 0.00) / 0.50 + 1 = 10

// グローバル変数（プログレス表示用）
int g_completed = 0;
int g_total = 0;
double g_start_time = 0;

// HSV to RGB変換（H: 0-1, S=1, V=1）
void hue_to_rgb(double h, uint8_t *r, uint8_t *g, uint8_t *b) {
    h = fmod(h, 1.0);
    if (h < 0) h += 1.0;
    
    double h6 = h * 6.0;
    int hi = (int)h6;
    double f = h6 - hi;
    
    switch (hi % 6) {
        case 0: *r = 255; *g = (uint8_t)(255 * f); *b = 0; break;
        case 1: *r = (uint8_t)(255 * (1 - f)); *g = 255; *b = 0; break;
        case 2: *r = 0; *g = 255; *b = (uint8_t)(255 * f); break;
        case 3: *r = 0; *g = (uint8_t)(255 * (1 - f)); *b = 255; break;
        case 4: *r = (uint8_t)(255 * f); *g = 0; *b = 255; break;
        case 5: *r = 255; *g = 0; *b = (uint8_t)(255 * (1 - f)); break;
    }
}

// 円（粒子）を描画
void draw_circle(uint8_t image[IMG_HEIGHT][IMG_WIDTH][3], int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= radius*radius) {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < IMG_WIDTH && py >= 0 && py < IMG_HEIGHT) {
                    image[py][px][0] = r;
                    image[py][px][1] = g;
                    image[py][px][2] = b;
                }
            }
        }
    }
}

// フレームを描画してPPMファイルに保存
void save_frame(const char *filename, double *psi_arr, double *rx_arr, double *ry_arr) {
    uint8_t image[IMG_HEIGHT][IMG_WIDTH][3];
    
    // 背景を黒でクリア
    memset(image, 0, sizeof(image));
    
    // 各粒子を描画
    for (int k = 0; k < N_PARTICLES; k++) {
        double norm_x = rx_arr[k] / (double)L;
        double norm_y = ry_arr[k] / (double)L;
        
        if (norm_x < 0) norm_x = 0;
        if (norm_x >= 1) norm_x = 0.999;
        if (norm_y < 0) norm_y = 0;
        if (norm_y >= 1) norm_y = 0.999;
        
        int px = (int)(norm_x * IMG_WIDTH);
        int py = (int)((1.0 - norm_y) * IMG_HEIGHT);
        
        double hue = fmod(psi_arr[k] / (2.0 * M_PI), 1.0);
        if (hue < 0) hue += 1.0;
        
        uint8_t r, g, b;
        hue_to_rgb(hue, &r, &g, &b);
        draw_circle(image, px, py, POINT_SIZE, r, g, b);
    }
    
    FILE *f = fopen(filename, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", IMG_WIDTH, IMG_HEIGHT);
        fwrite(image, sizeof(image), 1, f);
        fclose(f);
    }
}

// プログレスバーを表示
void print_progress(int completed, int total, double start_time) {
    double elapsed = omp_get_wtime() - start_time;
    double percent = 100.0 * completed / total;
    double rate = completed / elapsed;
    double remaining = (total - completed) / rate;
    
    // プログレスバー
    int bar_width = 40;
    int filled = (int)(bar_width * completed / total);
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %5.1f%% (%d/%d) ", percent, completed, total);
    
    // 経過時間と残り時間
    int elapsed_min = (int)(elapsed / 60);
    int elapsed_sec = (int)(elapsed) % 60;
    int remaining_min = (int)(remaining / 60);
    int remaining_sec = (int)(remaining) % 60;
    
    printf("Elapsed: %02d:%02d, ETA: %02d:%02d", 
           elapsed_min, elapsed_sec, remaining_min, remaining_sec);
    
    fflush(stdout);
}

// シミュレーションを実行してGIFを生成
void swarmOscillators(double c1, double c2, double c3, double alpha, int thread_id)
{
    // スレッドローカルな配列
    double psi[N_PARTICLES];
    double rx[N_PARTICLES];
    double ry[N_PARTICLES];
    double dpsi[N_PARTICLES];
    double drx[N_PARTICLES];
    double dry[N_PARTICLES];
    
    char gifname[200];
    char framedir[150];
    char framepath[200];
    char cmd[600];
    int frame_count = 0;
    
    // gifフォルダに出力
    sprintf(gifname, "gif/Swarm-c1_%.2f-c2_%.2f-c3_%.2f-alpha_%.2f.gif", c1, c2, c3, alpha);
    sprintf(framedir, "frames_%d", thread_id);
    
    // gifフォルダとフレームディレクトリを作成
    mkdir("gif", 0755);
    mkdir(framedir, 0755);
    
    // スレッド固有の乱数シード
    unsigned int seed = (unsigned int)(time(NULL) + thread_id * 1000);
    
    // 初期化
    for (int i = 0; i < N_PARTICLES; i++) {
        psi[i] = 6.28 * (double)rand_r(&seed) / RAND_MAX;
        rx[i] = (double)L * (double)rand_r(&seed) / RAND_MAX;
        ry[i] = (double)L * (double)rand_r(&seed) / RAND_MAX;
    }

    // シミュレーションループ
    for (int t = 0; t < LOOP; t++) {
        for (int i = 0; i < N_PARTICLES; i++) {
            dpsi[i] = 0;
            drx[i] = 0;
            dry[i] = 0;

            for (int j = 0; j < N_PARTICLES; j++) {
                if (i != j) {
                    double dx = rx[j] - rx[i];
                    double dy = ry[j] - ry[i];
                    
                    double r = sqrt(dx*dx + dy*dy);
                    double inv_r = 1.0 / (r + 1e-6);
                    double exp_r = exp(-r);
                    double phase_base = psi[j] - psi[i] + alpha * r;
                    double sin_psi = sin(phase_base - c1);
                    double sin_pos = sin(phase_base - c2);
                    
                    dpsi[i] += exp_r * sin_psi;
                    
                    double common = c3 * inv_r * exp_r * sin_pos;
                    drx[i] += dx * common;
                    dry[i] += dy * common;
                }
            }
        }
        
        for (int i = 0; i < N_PARTICLES; i++) {
            psi[i] = fmod(psi[i] + DT * dpsi[i] + 6.28, 6.28);
            rx[i] = fmod(rx[i] + DT * drx[i] + (double)L, (double)L);
            ry[i] = fmod(ry[i] + DT * dry[i] + (double)L, (double)L);
        }
        
        // フレームを保存
        if (t % FRAME_SKIP == 0) {
            sprintf(framepath, "%s/frame_%05d.ppm", framedir, frame_count);
            save_frame(framepath, psi, rx, ry);
            frame_count++;
        }
    }
    
    // ffmpegでGIFを生成
    sprintf(cmd, "ffmpeg -y -framerate 20 -i %s/frame_%%05d.ppm -vf \"palettegen\" %s/palette.png 2>/dev/null", framedir, framedir);
    system(cmd);
    
    sprintf(cmd, "ffmpeg -y -framerate 20 -i %s/frame_%%05d.ppm -i %s/palette.png -lavfi \"paletteuse\" %s 2>/dev/null", framedir, framedir, gifname);
    system(cmd);
    
    // フレームファイルを削除
    sprintf(cmd, "rm -rf %s", framedir);
    system(cmd);
}

int main(void)
{
    double params[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) {
        params[i] = PARAM_MIN + i * PARAM_STEP;
    }
    
    g_total = N_PARAMS * N_PARAMS * N_PARAMS * N_PARAMS;
    int n_threads = omp_get_max_threads();
    
    printf("===========================================\n");
    printf("  Swarm Oscillators Parameter Scan\n");
    printf("===========================================\n");
    printf("Parameters: c1, c2, c3, alpha\n");
    printf("Range: %.2f to %.2f (step %.2f)\n", PARAM_MIN, PARAM_MAX, PARAM_STEP);
    printf("Points per parameter: %d\n", N_PARAMS);
    printf("Total combinations: %d\n", g_total);
    printf("Threads: %d\n", n_threads);
    printf("===========================================\n\n");
    
    g_completed = 0;
    g_start_time = omp_get_wtime();
    
    #pragma omp parallel for collapse(4) schedule(dynamic)
    for (int i1 = 0; i1 < N_PARAMS; i1++) {
        for (int i2 = 0; i2 < N_PARAMS; i2++) {
            for (int i3 = 0; i3 < N_PARAMS; i3++) {
                for (int i4 = 0; i4 < N_PARAMS; i4++) {
                    double c1 = params[i1];
                    double c2 = params[i2];
                    double c3 = params[i3];
                    double alpha = params[i4];
                    
                    int thread_id = omp_get_thread_num();
                    swarmOscillators(c1, c2, c3, alpha, thread_id);
                    
                    #pragma omp atomic
                    g_completed++;
                    
                    // プログレス表示（100回に1回または完了時）
                    if (g_completed % 100 == 0 || g_completed == g_total) {
                        #pragma omp critical
                        {
                            print_progress(g_completed, g_total, g_start_time);
                        }
                    }
                }
            }
        }
    }
    
    printf("\n\n===========================================\n");
    printf("  All %d simulations completed!\n", g_total);
    double total_time = omp_get_wtime() - g_start_time;
    int total_min = (int)(total_time / 60);
    int total_sec = (int)(total_time) % 60;
    printf("  Total time: %02d:%02d\n", total_min, total_sec);
    printf("  Average: %.2f sec/simulation\n", total_time / g_total);
    printf("===========================================\n");
    
    return 0;
}
