# Izvoria mark

Two half-discs, offset: the app's icon and favicon. Colours are the app's accent (`theme.py`): `#6fbfa3` on dark
grounds, `#2d6a58` on light ones.

| File | Use |
|---|---|
| `izvoria-mark.svg` | The mark on a dark background (transparent around it) |
| `izvoria-mark-on-light.svg` | The mark on a light background |
| `izvoria-favicon.svg` | Browser tab: switches between the two colours with the system's light/dark setting |
| `izvoria-app-icon.svg` | The mark on a dark rounded square: the desktop window and taskbar icon (`ui/icons.py`, `app_icon`) |
| `izvoria-mark-<size>.png`, `izvoria-app-icon-<size>.png` | The same, rendered for places that take no SVG |

The web app serves copies from `web_deploy/frontend/public/assets/` (`favicon.svg`, and `apple-touch-icon.png`, the
180 px app icon). Change the SVGs here, then copy them there and re-render the PNGs (PySide6's `QSvgRenderer`).
