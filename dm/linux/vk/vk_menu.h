// The port's in-game options menu: F1, or Minus + Plus together on a controller, opens it over the game's picture (vk_window.c draws it; front_input.c
// routes the keys while it is open, and the game sees no input then). It is the port's own menu, not the game's Options screen: the game's files and save
// data are not touched. Device-free parts here (vk_menu.c): the model and its navigation, the settings file, and the menu's pixels.
//   Upscaler       Game (the game's own network) / FSR 1 / Linear: DM_UPSCALER's choices (vk_upscaler.h), applied from the next frame
//   Graphics tier  0 full size / 1 0.8x / 2 0.67x: DM_GFX_SCALE, read when the game starts, so a change applies on the next launch
// Choices are kept in a settings file, $DM_SETTINGS or $XDG_CONFIG_HOME/nxnative-asahi/settings.conf (~/.config when XDG_CONFIG_HOME is unset), as
// "upscaler=fsr1" and "gfx_scale=1" lines. An environment variable set at launch wins over the file.
#ifndef VK_MENU_H
#define VK_MENU_H
#include <stddef.h>
#include <stdint.h>

enum { VKX_MENU_UP = 1, VKX_MENU_DOWN, VKX_MENU_LEFT, VKX_MENU_RIGHT, VKX_MENU_ACCEPT, VKX_MENU_BACK };
enum { VKX_MENU_ROW_UPSCALER, VKX_MENU_ROW_TIER, VKX_MENU_ROW_CLOSE, VKX_MENU_ROWS };
enum { VKX_MENU_CHANGED_UPSCALER = 1, VKX_MENU_CHANGED_TIER = 2, VKX_MENU_CHANGED_OPEN = 4 };

typedef struct {
    int open, sel;
    int upscaler;       // VKX_UPSCALER_GAME, _FSR1 or _LINEAR
    int tier;           // the tier chosen, 0..2
    int tier_running;   // the tier this run started with (a different choice applies on the next launch)
} vkx_menu;

// Apply one action; returns the VKX_MENU_CHANGED_* bits it changed. Left and right step the selected row's choice (wrapping), accept steps it forward
// (or closes on the Close row), back closes
int vkx_menu_action(vkx_menu *m, int action);

// The settings file's text: sets *upscaler and *tier for the keys present and valid; returns 1 for an upscaler, 2 for a tier, ORed. Unknown keys and
// malformed lines are ignored (a later version's file still loads)
int vkx_menu_settings_parse(const char *text, int *upscaler, int *tier);
// The file's text for these choices; returns the length, or -1 when cap is too small
int vkx_menu_settings_format(char *out, size_t cap, int upscaler, int tier);

// The settings file's path: $DM_SETTINGS, else $XDG_CONFIG_HOME (or $HOME/.config)/nxnative-asahi/settings.conf; 0 when none can be named
int vkx_menu_settings_path(char *out, size_t cap);
// Read the file into *upscaler / *tier (vkx_menu_settings_parse); 0 when it is absent or empty
int vkx_menu_settings_load(int *upscaler, int *tier);
// Write the file (its directory made when missing); 1 on success
int vkx_menu_settings_save(int upscaler, int tier);

// The menu's size in pixels at an integer scale of the 8x8 font (scale 1 to 8)
void vkx_menu_size(unsigned scale, unsigned *w, unsigned *h);
// Draw the menu into w x h RGBA8 pixels (sRGB-encoded bytes, rows top first); w and h are what vkx_menu_size gave for this scale
void vkx_menu_draw(const vkx_menu *m, unsigned scale, uint8_t *rgba, unsigned w, unsigned h);
// The largest scale (8 at most) whose menu fits in 80% of a w x h picture's width and 60% of its height; 1 when none does
unsigned vkx_menu_scale_for(unsigned w, unsigned h);

#endif
