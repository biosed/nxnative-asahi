#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_menu.h"
#include "vk_upscaler.h"

static int bad;

static void check(int condition, const char *what)
{
    if (!condition) {
        printf("FAIL: %s\n", what);
        bad++;
    }
}

int main(int argc, char **argv)
{
    vkx_menu m = {.open = 1, .sel = VKX_MENU_ROW_UPSCALER, .upscaler = VKX_UPSCALER_GAME, .tier = 1, .tier_running = 1};
    check(vkx_menu_action(&m, VKX_MENU_RIGHT) == VKX_MENU_CHANGED_UPSCALER && m.upscaler == VKX_UPSCALER_FSR1, "right steps Game to FSR 1");
    check(vkx_menu_action(&m, VKX_MENU_ACCEPT) == VKX_MENU_CHANGED_UPSCALER && m.upscaler == VKX_UPSCALER_LINEAR, "accept steps FSR 1 to Linear");
    check(vkx_menu_action(&m, VKX_MENU_RIGHT) && m.upscaler == VKX_UPSCALER_GAME, "the upscaler choice wraps to Game");
    check(vkx_menu_action(&m, VKX_MENU_LEFT) && m.upscaler == VKX_UPSCALER_LINEAR, "left wraps Game back to Linear");
    check(vkx_menu_action(&m, VKX_MENU_DOWN) == 0 && m.sel == VKX_MENU_ROW_TIER, "down selects the tier row");
    check(vkx_menu_action(&m, VKX_MENU_RIGHT) == VKX_MENU_CHANGED_TIER && m.tier == 2, "right steps tier 1 to 2");
    check(vkx_menu_action(&m, VKX_MENU_RIGHT) && m.tier == 0, "the tier wraps to 0");
    check(m.tier_running == 1, "the running tier is unchanged by a choice");
    check(vkx_menu_action(&m, VKX_MENU_UP) == 0 && m.sel == VKX_MENU_ROW_UPSCALER, "up returns to the upscaler row");
    check(vkx_menu_action(&m, VKX_MENU_UP) == 0 && m.sel == VKX_MENU_ROW_CLOSE, "up from the first row wraps to Close");
    check(vkx_menu_action(&m, VKX_MENU_LEFT) == 0, "left on Close changes nothing");
    check(vkx_menu_action(&m, VKX_MENU_ACCEPT) == VKX_MENU_CHANGED_OPEN && !m.open, "accept on Close closes");
    check(vkx_menu_action(&m, VKX_MENU_RIGHT) == 0, "a closed menu takes no action");
    m.open = 1; m.sel = VKX_MENU_ROW_TIER;
    check(vkx_menu_action(&m, VKX_MENU_BACK) == VKX_MENU_CHANGED_OPEN && !m.open, "back closes from any row");

    int up = -1, tier = -1;
    check(vkx_menu_settings_parse("upscaler=fsr1\ngfx_scale=2\n", &up, &tier) == 3 && up == VKX_UPSCALER_FSR1 && tier == 2, "a full file parses");
    up = tier = -1;
    check(vkx_menu_settings_parse("# comment\r\nupscaler=linear \r\n", &up, &tier) == 1 && up == VKX_UPSCALER_LINEAR && tier == -1, "CRLF, trailing space and comments");
    up = tier = -1;
    check(vkx_menu_settings_parse("upscaler=probe\ngfx_scale=3\ngfx_scale=1x\nfuture=1\n", &up, &tier) == 0 && up == -1 && tier == -1,
          "probe, an out-of-range tier, a partial number and unknown keys are ignored");
    check(vkx_menu_settings_parse("", &up, &tier) == 0, "an empty file sets nothing");
    char buf[256];
    int n = vkx_menu_settings_format(buf, sizeof buf, VKX_UPSCALER_FSR1, 1);
    up = tier = -1;
    check(n > 0 && vkx_menu_settings_parse(buf, &up, &tier) == 3 && up == VKX_UPSCALER_FSR1 && tier == 1, "a written file reads back");
    check(vkx_menu_settings_format(buf, 8, VKX_UPSCALER_GAME, 0) == -1, "a too-small buffer is refused");

    char dir[] = "/tmp/t_vk_menu.XXXXXX", path[256];
    if (mkdtemp(dir)) {
        snprintf(path, sizeof path, "%s/a/b/settings.conf", dir);
        setenv("DM_SETTINGS", path, 1);
        up = tier = -1;
        check(vkx_menu_settings_load(&up, &tier) == 0, "a missing file loads nothing");
        check(vkx_menu_settings_save(VKX_UPSCALER_LINEAR, 2), "the file is written, its directories made");
        check(vkx_menu_settings_load(&up, &tier) == 3 && up == VKX_UPSCALER_LINEAR && tier == 2, "the written file loads back");
        check(vkx_menu_settings_save(VKX_UPSCALER_GAME, 0) && vkx_menu_settings_load(&up, &tier) == 3 && up == VKX_UPSCALER_GAME && tier == 0, "a rewrite replaces it");
        char cmd[300]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {}
        unsetenv("DM_SETTINGS");
    } else check(0, "a temporary directory");
    setenv("XDG_CONFIG_HOME", "/cfg", 1);
    check(vkx_menu_settings_path(path, sizeof path) && !strcmp(path, "/cfg/nxnative-asahi/settings.conf"), "XDG_CONFIG_HOME names the file");
    unsetenv("XDG_CONFIG_HOME"); setenv("HOME", "/home/u", 1);
    check(vkx_menu_settings_path(path, sizeof path) && !strcmp(path, "/home/u/.config/nxnative-asahi/settings.conf"), "without XDG_CONFIG_HOME it is under ~/.config");

    unsigned w1, h1, w3, h3;
    vkx_menu_size(1, &w1, &h1); vkx_menu_size(3, &w3, &h3);
    check(w3 == 3 * w1 && h3 == 3 * h1, "the size scales with the font");
    check(vkx_menu_scale_for(2560, 1440) >= 3, "a 1440p picture gets a readable scale");
    unsigned s = vkx_menu_scale_for(640, 360), sw, sh; vkx_menu_size(s, &sw, &sh);
    check(s == 1 || (sw * 10 <= 640 * 8 && sh * 10 <= 360 * 6), "the chosen scale fits");

    m = (vkx_menu){.open = 1, .sel = VKX_MENU_ROW_UPSCALER, .upscaler = VKX_UPSCALER_FSR1, .tier = 2, .tier_running = 1};
    uint8_t *px = calloc((size_t)w3 * h3, 4);
    vkx_menu_draw(&m, 3, px, w3, h3);
    size_t opaque = 0, sel = 0;
    for (size_t i = 0; i < (size_t)w3 * h3; i++) { opaque += px[4 * i + 3] == 255; sel += px[4 * i] == 230 && px[4 * i + 1] == 160 && px[4 * i + 2] == 40; }
    check(opaque == (size_t)w3 * h3, "every pixel of the menu is drawn opaque");
    check(sel > 0, "the selected row is highlighted");
    if (argc > 1) {   // t_vk_menu <out.ppm>: the menu as a picture, to look at
        FILE *f = fopen(argv[1], "wb");
        if (f) { fprintf(f, "P6\n%u %u\n255\n", w3, h3); for (size_t i = 0; i < (size_t)w3 * h3; i++) fwrite(px + 4 * i, 1, 3, f); fclose(f); }
    }
    free(px);

    printf("t_vk_menu: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
