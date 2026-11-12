# font8x8 (basic Latin)

`font8x8_basic.h` is copied unchanged from Daniel Hepper's font8x8,
<https://github.com/dhepper/font8x8> (`master`, file `font8x8_basic.h`), an
8x8 bitmap font for U+0000-U+007F derived from Marcel Sondaar's font8x8 and
the IBM public-domain VGA fonts.

- Licence: public domain (stated in the file header and the upstream README).
- SHA-256: `49d8df366296b203ca3211bc0672cf2a762135bf12710735b6292756b19dffd5`
- Rebuild/refresh: `curl -fL https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_basic.h`
  and check the digest above.

Used by `dm/linux/vk/vk_menu.c` to draw the port's in-game options menu. Each
glyph is eight row bytes, top row first; bit 0 of a row is its leftmost pixel.
