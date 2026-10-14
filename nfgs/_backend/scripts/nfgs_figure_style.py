"""Shared matplotlib font sizes for NFGS paper figures.

Base sizes match Figure 4 on its own canvas. Other figures that are placed
on a wider canvas, or are height-limited in the manuscript, should multiply
these sizes by ``font_scale_to_match_figure4`` so the glyphs have the same
size on the PDF page.
"""

FIG_TITLE_FONTSIZE = 12.0
FIG_ROW_LABEL_FONTSIZE = 12.0
FIG_AXIS_LABEL_FONTSIZE = 11.0
FIG_TICK_FONTSIZE = 10.0
FIG_CBAR_LABEL_FONTSIZE = 11.0
FIG_LEGEND_FONTSIZE = 10.0
FIG_ANNOTATION_FONTSIZE = 10.0
FIG_PANEL_LETTER_FONTSIZE = 12.0

# AASTeX ``manuscript`` page and the Figure 4 inclusion used as the reference.
PAPER_TEXTWIDTH_IN = 7.1
PAPER_TEXTHEIGHT_IN = 9.09
FIG4_CANVAS_WIDTH_IN = 9.50
FIG4_WIDTH_FRAC = 0.98


def font_scale_to_match_figure4(
    canvas_width_in: float,
    canvas_height_in: float | None = None,
    *,
    width_frac: float = 1.0,
    height_frac: float | None = None,
) -> float:
    """Return the factor that makes on-page text match Figure 4.

    ``width_frac`` and ``height_frac`` are the ``\\textwidth`` and
    ``\\textheight`` fractions in ``\\includegraphics``. When both are set,
    the more restrictive one is used, matching ``keepaspectratio``.
    """
    reference_scale = (
        FIG4_WIDTH_FRAC * PAPER_TEXTWIDTH_IN / FIG4_CANVAS_WIDTH_IN
    )
    display_scale = width_frac * PAPER_TEXTWIDTH_IN / float(canvas_width_in)
    if height_frac is not None and canvas_height_in is not None:
        height_scale = height_frac * PAPER_TEXTHEIGHT_IN / float(canvas_height_in)
        display_scale = min(display_scale, height_scale)
    if display_scale <= 0.0:
        raise ValueError("display scale must be positive")
    return reference_scale / display_scale
