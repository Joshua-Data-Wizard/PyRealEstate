"""
The shared look of pyRealEstate's charts (private; used by RealEstateMetrics
and Time_Trending).

Every chart has a warm off-white background, light gridlines, no top or right
frame, muted tick labels, and a bold title with a short subtitle above the
plot. Data is drawn in blue (and orange for a second series): a colorblind-
safe pair from a validated reference palette. Everything that gives context
rather than data (gridlines, reference lines, other groups) is gray, so the
main message stands out.

matplotlib is imported only inside the one function that makes a figure, so
importing this module stays fast (see the top of RealEstateMetrics.py).
"""

import textwrap
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:  # for editors and type checkers only; never imported at run time
    import matplotlib.axes

_PLOT_STYLE = {
    "surface": "#fcfcfb",         # figure and plot background
    "text": "#0b0b0b",            # titles and key numbers
    "text_secondary": "#52514e",  # subtitles, axis labels, tick labels
    "muted": "#898781",           # notes, tick marks, labels that explain the chart
    "grid": "#e1e0d9",            # gridlines
    "axis": "#c3c2b7",            # the frame of the plot, and reference lines
    "blue": "#2a78d6",            # the first data color
    "blue_dark": "#1c5cab",       # the same blue, darker: points and medians drawn on blue
    "orange": "#eb6834",          # the second data color
    "band": "#f0efec",            # a shaded reference range
    "context": "#b9b8b1",         # data shown for context, not the main message
    "context_dark": "#6f6e69",    # outlines and medians of context data
}

# The minus sign used in chart labels. A hyphen (-) is shorter and easy to miss
# next to a number; matplotlib's own tick labels use this sign too.
_MINUS = "\u2212"


def _number(value: float, spec: str = ".2f", suffix: str = "") -> str:
    """
    Format a number for a chart label, with a true minus sign.

    ``_number(-12.494, ".2f", "%")`` gives "\u221212.49%";
    ``_number(10, "+.0f", "%")`` gives "+10%".
    """
    return f"{value:{spec}}".replace("-", _MINUS) + suffix


def _compact_dollars(value: float) -> str:
    """
    A dollar amount in three significant digits: $950, $87.5K, $1.25M.

    A value that rounds up to 1,000 of one unit is shown in the next unit
    ($999,600 is "$1M", not "$1e+03K").
    """
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= size * 0.9995:
            return f"${value / size:.3g}{suffix}"
    return f"${value:,.0f}"


def _new_axes(figsize: Tuple[float, float]) -> "matplotlib.axes.Axes":
    """A new figure with one plot in the library's colors; returns its axes."""
    import matplotlib.pyplot as plt  # imported here: slow to import

    _, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.figure.set_facecolor(_PLOT_STYLE["surface"])
    ax.set_facecolor(_PLOT_STYLE["surface"])
    return ax


def _style_axes(ax: "matplotlib.axes.Axes", grid_axis: Optional[str] = "y") -> None:
    """
    Light gridlines (``grid_axis`` "x", "y", "both", or None for none), no
    top or right frame, and muted ticks.
    """
    style = _PLOT_STYLE
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=style["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style["axis"])
    ax.tick_params(colors=style["muted"], labelcolor=style["text_secondary"], labelsize=9)


def _join_with_dots(parts: List[str], width: int = 100) -> str:
    """Join parts with ' · ', starting a new line before the text gets wider than ``width``."""
    lines, current = [], ""
    for part in parts:
        candidate = f"{current}  ·  {part}" if current else part
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def _plot_width_inches(ax: "matplotlib.axes.Axes", own_figure: bool) -> float:
    """
    The width of the plot, in inches. A figure made by the library
    (``own_figure``) uses nearly its full width; on the caller's axes, the
    width of those axes is used.
    """
    plot_share = 0.9 if own_figure else ax.get_position().width
    return plot_share * ax.figure.get_figwidth()


def _fit_title(
    ax: "matplotlib.axes.Axes", own_figure: bool, title: str, subtitle_parts: List[str]
) -> Tuple[str, str]:
    """
    Wrap a title, and join the subtitle's parts, to the width of the plot.

    At their font sizes, the subtitle fits about 13 characters per inch and
    the title about 9, so both also fit a narrow panel of a larger figure.
    """
    plot_inches = _plot_width_inches(ax, own_figure)
    subtitle = _join_with_dots(subtitle_parts, width=max(40, int(plot_inches * 13)))
    title = textwrap.fill(title, width=max(30, int(plot_inches * 9)))
    return title, subtitle


def _add_title(
    ax: "matplotlib.axes.Axes",
    title: str,
    subtitle: Optional[str] = None,
    offset: float = 4,
    title_size: float = 13,
) -> None:
    """
    Write the subtitle, then the bold title above it, left-aligned above the
    plot. ``offset`` (in points) leaves room below them, e.g. for a legend
    placed above the plot.
    """
    style = _PLOT_STYLE
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, offset),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9.5, color=style["text_secondary"])
        offset += 13 * (subtitle.count("\n") + 1) + 6
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, offset),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=title_size, fontweight="bold", color=style["text"])
