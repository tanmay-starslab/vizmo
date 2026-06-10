"""Design tokens for the vizmo dark theme.

Single source of truth for panel colors and geometry. Panels migrate
to these tokens incrementally; new UI code must use them.
"""


class DarkTheme:
    # Background
    BG_DEEP = (10, 13, 18, 255)
    BG_SURFACE = (19, 24, 31, 235)
    BG_RAISED = (26, 33, 42, 255)
    # Borders
    BORDER = (30, 39, 48, 255)
    BORDER_FOCUS = (61, 126, 255, 255)
    # Accent
    ACCENT = (61, 126, 255, 255)
    ACCENT_DIM = (30, 63, 127, 255)
    # Text
    TEXT_PRIMARY = (232, 237, 243, 255)
    TEXT_SECONDARY = (138, 150, 166, 255)
    TEXT_DISABLED = (72, 80, 90, 255)
    # Status
    SUCCESS = (39, 201, 122, 255)
    WARNING = (245, 166, 35, 255)
    DANGER = (229, 62, 62, 255)
    INFO = (61, 126, 255, 255)
    # Geometry
    CORNER_RADIUS = 6
    PADDING = 8
    ITEM_HEIGHT = 22
    FONT_SIZE_SM = 11
    FONT_SIZE_MD = 13
    FONT_SIZE_LG = 15

    @classmethod
    def colors(cls):
        return {k: v for k, v in vars(cls).items()
                if k.isupper() and isinstance(v, tuple)}
