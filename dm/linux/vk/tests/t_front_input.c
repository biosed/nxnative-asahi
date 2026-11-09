#include "dm.h"
#include "dm_input.h"

static uint64_t g_buttons;
static int32_t g_axes[4];
static int g_updates;

void dm_input_set(uint64_t buttons, int32_t lx, int32_t ly, int32_t rx, int32_t ry) {
    g_buttons = buttons;
    g_axes[0] = lx; g_axes[1] = ly; g_axes[2] = rx; g_axes[3] = ry;
    g_updates++;
}
void dm_pad_set_player(int player, uint64_t buttons, int32_t lx, int32_t ly, int32_t rx, int32_t ry) {
    (void)player; (void)buttons; (void)lx; (void)ly; (void)rx; (void)ry;
}
void dm_pad_players(uint32_t mask) { (void)mask; }
uint32_t dm_player_npad(int player) { return (uint32_t)player; }

#include "../front_input.c"

static int failures;
#define CHECK(c, what) do { if (!(c)) { LOG("[RED] front input test: %s", what); failures++; } } while (0)

static void key(SDL_Scancode scancode, bool down) {
    SDL_KeyboardEvent event = {.scancode = scancode, .down = down};
    input_key(&event);
}

int main(int argc, char **argv) {
    if (argc != 2) die("front input test: usage: t_front_input <map>");
    if (setenv("DM_INPUT_MAP", argv[1], 1)) die("front input test: setenv failed");
    input_init();

    key(SDL_SCANCODE_RETURN, true);
    CHECK(g_updates == 1 && g_buttons == DM_BTN_A, "Return maps to A down");
    key(SDL_SCANCODE_RETURN, false);
    CHECK(g_updates == 2 && g_buttons == 0, "Return maps to A up");

    key(SDL_SCANCODE_P, true);
    CHECK(g_buttons == DM_BTN_PLUS, "P maps to Plus");
    key(SDL_SCANCODE_P, false);

    key(SDL_SCANCODE_UP, true);
    CHECK(g_buttons == DM_BTN_UP && g_axes[0] == 0 && g_axes[1] == DM_STICK_MAX,
          "Up maps to d-pad Up and left-stick Up");
    key(SDL_SCANCODE_UP, false);
    CHECK(g_buttons == 0 && g_axes[0] == 0 && g_axes[1] == 0, "Up release clears both mappings");

    key(SDL_SCANCODE_ESCAPE, true);
    CHECK(g_buttons == DM_BTN_B, "Escape maps to B");
    key(SDL_SCANCODE_ESCAPE, false);
    CHECK(g_buttons == 0, "Escape release clears B");

    LOG("front input test: %s (%d failures)", failures ? "FAILED" : "passed", failures);
    return failures != 0;
}
