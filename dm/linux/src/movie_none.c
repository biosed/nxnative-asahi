// First-runtime Linux scaffold.  The native host and graphics bring-up do not
// depend on an Apple or Android codec API.  Keeping unsupported paths visible
// is preferable to returning fake frames or silently pretending decode
// succeeded.  AV1 is implemented separately by hle_movie_av1.c.
#include "dm.h"
#include "hle_movie_avc.h"
#include <arm_neon.h>
#include "vpx/vpx_image.h"

struct mavc { int red, rgba; };

int dm_movie_playing(void) { return 0; }

mavc *mavc_new(void) { mavc *a = calloc(1, sizeof *a); if (!a) die("movie: AVC decoder allocation failed"); return a; }
void mavc_free(mavc *a) { free(a); }
void mavc_param_sets(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
void mavc_avcc(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
int mavc_decode(mavc *a, const uint8_t *p, int n, int64_t pts) {
    (void)p; (void)n; (void)pts;
    if (!a->red++) LOG("[RED] movie: Linux FFmpeg AVC bridge not linked yet");
    return -1;
}
int mavc_take(mavc *a, int flush, uint8_t *out, int w, int h, int64_t *pts) { (void)a; (void)flush; (void)out; (void)w; (void)h; (void)pts; return 0; }
int mavc_queued(const mavc *a) { (void)a; return 0; }
int mavc_bt709(const mavc *a, int h) { (void)a; return h >= 720; }
void mavc_set_rgba(mavc *a) { a->rgba = 1; }
void mavc_drop(mavc *a) { (void)a; }

// The full NintendoSDK_movie decoder is not linked on Linux yet, but
// nn::codec's AV1/VP9 takeover uses these shared I420 conversion helpers.  The
// old scaffold returned synthetic black here, discarding correctly decoded
// pictures before the renderer ever saw them.
void md_nv12(const vpx_image_t *im, uint8_t *dst, int w, int h) {
    int cw = (int)im->d_w < w ? (int)im->d_w : w, ch = (int)im->d_h < h ? (int)im->d_h : h;
    if (cw < w || ch < h) { memset(dst, 16, (size_t)w * h); memset(dst + (size_t)w * h, 128, (size_t)w * h / 2); }
    for (int y = 0; y < ch; y++) memcpy(dst + (size_t)y * w, im->planes[0] + (size_t)y * im->stride[0], cw);
    int cw2 = (cw + 1) / 2, ch2 = (ch + 1) / 2;
    for (int y = 0; y < ch2 && y < h / 2; y++) {
        const uint8_t *u = im->planes[1] + (size_t)y * im->stride[1], *v = im->planes[2] + (size_t)y * im->stride[2];
        uint8_t *o = dst + (size_t)w * h + (size_t)y * w;
        int x = 0;
        for (; x + 16 <= cw2; x += 16) { uint8x16x2_t t = {{vld1q_u8(u + x), vld1q_u8(v + x)}}; vst2q_u8(o + 2 * x, t); }
        for (; x < cw2 && 2 * x + 1 < w; x++) { o[2 * x] = u[x]; o[2 * x + 1] = v[x]; }
    }
}

void md_rgba(const vpx_image_t *im, uint8_t *dst, int w, int h) {
    int full = im->range == VPX_CR_FULL_RANGE, b709 = im->cs == VPX_CS_BT_709 || (im->cs == VPX_CS_UNKNOWN && h >= 720);
    double ky = full ? 1.0 : 255.0 / 219.0, kc = full ? 1.0 : 255.0 / 224.0;
    double rv = b709 ? 1.5748 : 1.402, gu = b709 ? 0.1873 : 0.3441, gv = b709 ? 0.4681 : 0.7141, bu = b709 ? 1.8556 : 1.772;
    int Y = (int)(ky * 65536), RV = (int)(rv * kc * 65536), GU = (int)(gu * kc * 65536), GV = (int)(gv * kc * 65536), BU = (int)(bu * kc * 65536), y0 = full ? 0 : 16;
    int cw = (int)im->d_w < w ? (int)im->d_w : w, ch = (int)im->d_h < h ? (int)im->d_h : h;
    if (cw < w || ch < h) for (size_t i = 0; i < (size_t)w * h; i++) ((uint32_t *)dst)[i] = 0xff000000u;
    #define CL(v) ((v) < 0 ? 0 : (v) > 255 ? 255 : (v))
    for (int y = 0; y < ch; y++) {
        const uint8_t *py = im->planes[0] + (size_t)y * im->stride[0], *pu = im->planes[1] + (size_t)(y >> 1) * im->stride[1], *pv = im->planes[2] + (size_t)(y >> 1) * im->stride[2];
        uint8_t *o = dst + (size_t)y * w * 4;
        for (int x = 0; x < cw; x++) {
            int l = (py[x] - y0) * Y, u = pu[x >> 1] - 128, v = pv[x >> 1] - 128;
            int r = (l + RV * v + 32768) >> 16, g = (l - GU * u - GV * v + 32768) >> 16, b = (l + BU * u + 32768) >> 16;
            o[4 * x] = (uint8_t)CL(r); o[4 * x + 1] = (uint8_t)CL(g); o[4 * x + 2] = (uint8_t)CL(b); o[4 * x + 3] = 255;
        }
    }
    #undef CL
}
