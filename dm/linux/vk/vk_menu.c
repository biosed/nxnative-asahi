// The port's in-game options menu, device-free parts (vk_menu.h): model, settings file text and pixels.
#include "vk_menu.h"
#include "vk_upscaler.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "../../third_party/font8x8/font8x8_basic.h"

// The upscaler choices in the order left/right steps through them, and their labels
static const int UPSCALERS[] = {VKX_UPSCALER_GAME, VKX_UPSCALER_FSR1, VKX_UPSCALER_LINEAR};
static const char *const UPSCALER_LABEL[] = {"Game", "FSR 1", "Linear"};
static const char *const UPSCALER_KEY[] = {"game", "fsr1", "linear"};
static const char *const TIER_LABEL[] = {"0  full size", "1  0.8x", "2  0.67x"};
enum { NUPSCALERS = 3, NTIERS = 3 };

static int upscaler_index(int mode) {
    for (int i = 0; i < NUPSCALERS; i++) if (UPSCALERS[i] == mode) return i;
    return 0;
}

int vkx_menu_action(vkx_menu *m, int action) {
    if (!m->open) return 0;
    if (m->sel < 0 || m->sel >= VKX_MENU_ROWS) m->sel = 0;
    if (action == VKX_MENU_BACK || (action == VKX_MENU_ACCEPT && m->sel == VKX_MENU_ROW_CLOSE)) { m->open = 0; return VKX_MENU_CHANGED_OPEN; }
    if (action == VKX_MENU_UP) { m->sel = (m->sel + VKX_MENU_ROWS - 1) % VKX_MENU_ROWS; return 0; }
    if (action == VKX_MENU_DOWN) { m->sel = (m->sel + 1) % VKX_MENU_ROWS; return 0; }
    int step = action == VKX_MENU_LEFT ? -1 : action == VKX_MENU_RIGHT || action == VKX_MENU_ACCEPT ? 1 : 0;
    if (!step) return 0;
    if (m->sel == VKX_MENU_ROW_UPSCALER) { m->upscaler = UPSCALERS[(upscaler_index(m->upscaler) + NUPSCALERS + step) % NUPSCALERS]; return VKX_MENU_CHANGED_UPSCALER; }
    if (m->sel == VKX_MENU_ROW_TIER) { int t = m->tier >= 0 && m->tier < NTIERS ? m->tier : 0; m->tier = (t + NTIERS + step) % NTIERS; return VKX_MENU_CHANGED_TIER; }
    return 0;
}

int vkx_menu_settings_parse(const char *text, int *upscaler, int *tier) {
    int found = 0;
    for (const char *l = text; l && *l;) {
        const char *e = strchr(l, '\n'); size_t n = e ? (size_t)(e - l) : strlen(l);
        char line[128];
        if (n < sizeof line) {
            memcpy(line, l, n); line[n] = 0;
            while (n && (line[n - 1] == '\r' || line[n - 1] == ' ' || line[n - 1] == '\t')) line[--n] = 0;
            const char *v = strchr(line, '=');
            if (v && line[0] != '#') {
                size_t kn = (size_t)(v - line); v++;
                if (kn == 8 && !strncmp(line, "upscaler", 8)) {
                    for (int i = 0; i < NUPSCALERS; i++) if (!strcmp(v, UPSCALER_KEY[i])) { *upscaler = UPSCALERS[i]; found |= 1; }
                } else if (kn == 9 && !strncmp(line, "gfx_scale", 9) && v[0] >= '0' && v[0] < '0' + NTIERS && !v[1]) {
                    *tier = v[0] - '0'; found |= 2;
                }
            }
        }
        l = e ? e + 1 : NULL;
    }
    return found;
}

int vkx_menu_settings_format(char *out, size_t cap, int upscaler, int tier) {
    int t = tier >= 0 && tier < NTIERS ? tier : 0;
    int n = snprintf(out, cap, "# nxnative-asahi settings, written by the in-game options menu (F1); a launch variable overrides a line\nupscaler=%s\ngfx_scale=%d\n",
                     UPSCALER_KEY[upscaler_index(upscaler)], t);
    return n < 0 || (size_t)n >= cap ? -1 : n;
}

int vkx_menu_settings_path(char *out, size_t cap) {
    const char *e = getenv("DM_SETTINGS"), *x = getenv("XDG_CONFIG_HOME"), *h = getenv("HOME");
    int n = e && *e ? snprintf(out, cap, "%s", e)
          : x && *x ? snprintf(out, cap, "%s/nxnative-asahi/settings.conf", x)
          : h && *h ? snprintf(out, cap, "%s/.config/nxnative-asahi/settings.conf", h) : -1;
    return n > 0 && (size_t)n < cap;
}

int vkx_menu_settings_load(int *upscaler, int *tier) {
    char path[1024], text[4096];
    if (!vkx_menu_settings_path(path, sizeof path)) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(text, 1, sizeof text - 1, f); fclose(f); text[n] = 0;
    return vkx_menu_settings_parse(text, upscaler, tier);
}

// Make every missing directory above path's last component
static int make_parents(char *path) {
    for (char *p = path + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0; int ok = !mkdir(path, 0755) || errno == EEXIST; *p = '/';
        if (!ok) return 0;
    }
    return 1;
}

int vkx_menu_settings_save(int upscaler, int tier) {
    char path[1024], tmp[1100], text[512];
    int n = vkx_menu_settings_format(text, sizeof text, upscaler, tier);
    if (n < 0 || !vkx_menu_settings_path(path, sizeof path) || !make_parents(path)) return 0;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);   // written whole, then renamed over the old file: a crash leaves one or the other
    FILE *f = fopen(tmp, "w");
    if (!f) return 0;
    int ok = fwrite(text, 1, (size_t)n, f) == (size_t)n;
    ok = !fclose(f) && ok;
    return ok && !rename(tmp, path);
}

// Layout in 8x8 character cells: a box COLS wide and LINES tall, each text line LINE_PX font pixels high
enum { COLS = 38, LINES = 9, LINE_PX = 12, PAD_PX = 6, VALUE_COL = 19, VALUE_W = 14 };
typedef struct { uint8_t r, g, b; } rgb;
static const rgb BG = {38, 28, 18}, BORDER = {196, 152, 76}, TITLE = {240, 200, 110}, TEXT = {238, 226, 198}, DIM = {168, 148, 116},
                 SEL_BG = {230, 160, 40}, SEL_TEXT = {40, 28, 16}, RULE = {120, 92, 50};

void vkx_menu_size(unsigned scale, unsigned *w, unsigned *h) {
    if (scale < 1) scale = 1;
    if (scale > 8) scale = 8;
    *w = (COLS * 8 + 2 * PAD_PX) * scale; *h = (LINES * LINE_PX + 2 * PAD_PX) * scale;
}

unsigned vkx_menu_scale_for(unsigned w, unsigned h) {
    for (unsigned s = 8; s > 1; s--) { unsigned mw, mh; vkx_menu_size(s, &mw, &mh); if (mw * 10 <= w * 8 && mh * 10 <= h * 6) return s; }
    return 1;
}

typedef struct { uint8_t *p; unsigned w, h, s; } canvas;
static void fill(const canvas *c, unsigned x, unsigned y, unsigned w, unsigned h, rgb col) {
    for (unsigned j = y; j < y + h && j < c->h; j++)
        for (unsigned i = x; i < x + w && i < c->w; i++) { uint8_t *q = c->p + ((size_t)j * c->w + i) * 4; q[0] = col.r; q[1] = col.g; q[2] = col.b; q[3] = 255; }
}
// Text at a character column and line, in font pixels scaled by c->s
static void text(const canvas *c, unsigned col, unsigned line, const char *s, rgb fg) {
    unsigned x0 = (PAD_PX + col * 8) * c->s, y0 = (PAD_PX + line * LINE_PX + (LINE_PX - 8) / 2) * c->s;
    for (unsigned k = 0; s[k] && col + k < COLS; k++) {
        unsigned char ch = (unsigned char)s[k]; if (ch >= 128) ch = '?';
        for (unsigned r = 0; r < 8; r++) {
            unsigned bits = (unsigned char)font8x8_basic[ch][r];
            for (unsigned b = 0; b < 8; b++) if (bits >> b & 1) fill(c, x0 + (k * 8 + b) * c->s, y0 + r * c->s, c->s, c->s, fg);
        }
    }
}
static void row(const canvas *c, const vkx_menu *m, int which, unsigned line, const char *label, const char *value) {
    int sel = m->sel == which;
    if (sel) fill(c, (PAD_PX + 8) * c->s, (PAD_PX + line * LINE_PX) * c->s, (COLS - 2) * 8 * c->s, LINE_PX * c->s, SEL_BG);
    rgb fg = sel ? SEL_TEXT : TEXT;
    text(c, 2, line, label, fg);
    if (value) {
        char v[COLS + 1]; snprintf(v, sizeof v, "< %-*.*s >", VALUE_W, VALUE_W, value);
        text(c, VALUE_COL, line, v, fg);
    }
}

void vkx_menu_draw(const vkx_menu *m, unsigned scale, uint8_t *rgba, unsigned w, unsigned h) {
    if (scale < 1) scale = 1;
    if (scale > 8) scale = 8;
    canvas c = {rgba, w, h, scale};
    fill(&c, 0, 0, w, h, BORDER);
    fill(&c, 2 * scale, 2 * scale, w > 4 * scale ? w - 4 * scale : 0, h > 4 * scale ? h - 4 * scale : 0, BG);
    const char *title = "Port Options";
    text(&c, (COLS - (unsigned)strlen(title)) / 2, 0, title, TITLE);
    fill(&c, (PAD_PX + 8) * scale, (PAD_PX + LINE_PX + LINE_PX / 2) * scale, (COLS - 2) * 8 * scale, scale, RULE);
    row(&c, m, VKX_MENU_ROW_UPSCALER, 2, "Upscaler", UPSCALER_LABEL[upscaler_index(m->upscaler)]);
    row(&c, m, VKX_MENU_ROW_TIER, 3, "Graphics tier", TIER_LABEL[m->tier >= 0 && m->tier < NTIERS ? m->tier : 0]);
    if (m->tier != m->tier_running) text(&c, 2, 4, "The tier change applies next launch", DIM);
    row(&c, m, VKX_MENU_ROW_CLOSE, 5, "Close", NULL);
    fill(&c, (PAD_PX + 8) * scale, (PAD_PX + 6 * LINE_PX + LINE_PX / 2) * scale, (COLS - 2) * 8 * scale, scale, RULE);
    text(&c, 2, 7, "A/Enter: change   B/Esc/F1: close", DIM);
}
