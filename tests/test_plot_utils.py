import textwrap

import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from gabriel.utils.plot_utils import bar_plot, box_plot, line_plot, regression_plot


@pytest.fixture(autouse=True)
def close_plots():
    try:
        yield
    finally:
        plt.close("all")


def _extract_bar_values():
    fig_numbers = plt.get_fignums()
    assert fig_numbers, "bar_plot should create at least one figure"
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.containers, "plot should contain bar containers"
    container = ax.containers[0]
    return list(container.datavalues)


def test_bar_plot_counts_defaults_to_frequencies():
    df = pd.DataFrame(
        {
            "fruit": [
                "apple",
                "banana",
                "apple",
                "apple",
                "carrot",
                "banana",
                "banana",
            ]
        }
    )

    bar_plot(data=df, category_column="fruit")

    observed = _extract_bar_values()
    expected = list(df["fruit"].astype(str).value_counts())
    assert observed == expected


def test_bar_plot_counts_as_percentages():
    df = pd.DataFrame(
        {
            "pet": [
                "cat",
                "dog",
                "cat",
                "cat",
                "parrot",
                "dog",
            ]
        }
    )

    bar_plot(data=df, category_column="pet", as_percent=True)

    observed = _extract_bar_values()
    counts = df["pet"].astype(str).value_counts()
    expected = list((counts / counts.sum()) * 100)
    assert pytest.approx(observed, rel=1e-6) == expected


def test_bar_plot_counts_respects_category_cap():
    df = pd.DataFrame({"label": [f"item_{idx % 20}" for idx in range(60)]})

    bar_plot(data=df, category_column="label", category_cap=5)

    observed = _extract_bar_values()
    counts = df["label"].astype(str).value_counts().head(5)
    assert len(observed) == 5
    assert observed == list(counts)


def test_bar_plot_horizontal_descending_places_largest_at_top():
    df = pd.DataFrame(
        {
            "label": ["alpha", "beta", "gamma", "beta"],
            "value": [2, 5, 1, 3],
        }
    )

    bar_plot(
        data=df,
        category_column="label",
        value_column="value",
        orientation="horizontal",
    )

    fig_numbers = plt.get_fignums()
    assert fig_numbers, "bar_plot should create at least one figure"
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.yaxis_inverted(), "horizontal charts should display largest values at the top"


def test_bar_plot_vertical_auto_size_prefers_wider_layout():
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e", "f"],
            "score": [1, 2, 3, 4, 5, 6],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score")

    fig_numbers = plt.get_fignums()
    assert fig_numbers, "auto-sized bar plot should create a figure"
    fig = plt.figure(fig_numbers[0])
    assert fig.get_figwidth() >= fig.get_figheight(), "auto sizing should prefer wider-than-tall figures"


def test_bar_plot_long_labels_expand_figure_width():
    long_label = "This is an extremely long feature description that should wrap across multiple lines"
    df = pd.DataFrame(
        {
            "feature": [long_label + str(idx) for idx in range(3)],
            "score": [1.0, 2.0, 3.0],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score")

    fig_numbers = plt.get_fignums()
    assert fig_numbers
    fig = plt.figure(fig_numbers[0])
    assert fig.get_figwidth() > 13.0, "long labels should trigger a wider figure"


def test_bar_plot_auto_size_limits_width_for_small_counts():
    long_label = "This is a deliberately verbose label that previously forced absurd widths"
    df = pd.DataFrame(
        {
            "feature": [f"{long_label} {idx}" for idx in range(6)],
            "score": [float(idx) for idx in range(6)],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score")

    fig_numbers = plt.get_fignums()
    assert fig_numbers
    fig = plt.figure(fig_numbers[0])
    assert fig.get_figwidth() <= 24.0, "small bar counts should not explode the figure width"


def test_bar_plot_wrapped_story_style_labels_prefer_balanced_widths():
    features = [
        "Winning arguments are longer, syntactically complex, and deployed more than losing arguments",
        "Losing arguments are concise, single focused, corrective, or terse compared with persuasive replies",
        "Winning arguments hedge, qualify, and display nuance more than losing arguments",
        "Winning arguments deploy hypotheticals, scenarios, and slippery slope thought experiments",
        "Winning arguments use concrete examples far more than losing arguments",
        "Winning arguments employ formal, philosophical, legal, or logical reasoning more than losing arguments",
    ]
    df = pd.DataFrame(
        {
            "feature": features,
            "score": [36.9, 23.8, 17.5, 17.1, 16.8, 15.7],
        }
    )

    bar_plot(
        data=df,
        category_column="feature",
        value_column="score",
        as_percent=True,
    )

    fig_numbers = plt.get_fignums()
    assert fig_numbers
    fig = plt.figure(fig_numbers[0])
    width = fig.get_figwidth()
    assert 13.5 <= width <= 18.5, "long narrative labels should stay in a reviewable range"
    tick_texts = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert any("\n" in text for text in tick_texts if text)


def test_bar_plot_label_wrap_mode_none_disables_wrapping():
    df = pd.DataFrame(
        {
            "feature": [
                "A fairly long feature name with spaces",
                "Another reasonably descriptive label",
            ],
            "score": [1.0, 2.0],
        }
    )

    bar_plot(
        data=df,
        category_column="feature",
        value_column="score",
        label_wrap_mode="none",
    )

    fig_numbers = plt.get_fignums()
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    tick_texts = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert all("\n" not in text for text in tick_texts if text)


def test_bar_plot_label_wrap_mode_fixed_honours_wrap_width():
    df = pd.DataFrame(
        {
            "feature": ["alpha beta gamma delta epsilon"],
            "score": [1.0],
        }
    )

    bar_plot(
        data=df,
        category_column="feature",
        value_column="score",
        wrap_width=10,
        label_wrap_mode="fixed",
        min_wrap_chars=5,
    )

    fig_numbers = plt.get_fignums()
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    text = fig.axes[0].get_xticklabels()[0].get_text()
    assert text == textwrap.fill("alpha beta gamma delta epsilon", width=10)


def test_bar_plot_label_wrap_mode_rejects_invalid_values():
    df = pd.DataFrame({"feature": ["alpha"], "score": [1.0]})

    with pytest.raises(ValueError, match="label_wrap_mode"):
        bar_plot(
            data=df,
            category_column="feature",
            value_column="score",
            label_wrap_mode="diagonal",
        )


def test_bar_plot_aliases_x_label_font_size():
    df = pd.DataFrame(
        {
            "feature": ["alpha", "beta"],
            "score": [1.0, 2.0],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score", x_label_font_size=7)

    fig_numbers = plt.get_fignums()
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    ax = fig.axes[0]
    tick_sizes = {round(tick.get_fontsize()) for tick in ax.get_xticklabels() if tick.get_text()}
    assert 7 in tick_sizes


def test_bar_plot_removed_wrap_reference_param_raises():
    df = pd.DataFrame(
        {
            "feature": ["alpha", "beta"],
        }
    )

    with pytest.raises(TypeError, match="wrap_scale_reference has been removed"):
        bar_plot(data=df, category_column="feature", wrap_scale_reference=20)


def test_line_plot_accepts_multiple_y_columns_without_series():
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "alpha": [1.0, 3.0, 5.0, 7.0],
            "beta": [2.0, 4.0, 6.0, 8.0],
        }
    )

    figs_axes = line_plot(
        df,
        x="year",
        y=["alpha", "beta"],
        show=False,
        gradient=False,
        grid=False,
    )

    assert len(figs_axes) == 1
    fig, ax = figs_axes[0]
    series_data = {line.get_label(): line for line in ax.lines}
    assert set(series_data) >= {"alpha", "beta"}
    alpha_line = series_data["alpha"]
    beta_line = series_data["beta"]
    assert list(alpha_line.get_xdata()) == [2020, 2021]
    assert list(beta_line.get_xdata()) == [2020, 2021]
    assert alpha_line.get_ydata().tolist() == [2.0, 6.0]
    assert beta_line.get_ydata().tolist() == [3.0, 7.0]


@pytest.mark.parametrize("cmap", ["viridis", None, ListedColormap(["red", "green", "blue"])])
def test_bar_plot_preserves_colormap_inputs_and_samples(cmap, tmp_path):
    target = tmp_path / "bars.png"
    bar_plot(
        ["a", "b", "c"], [1, 2, 3], cmap=cmap,
        gradient_start=0, gradient_end=1, sort_mode=None,
        save_path=target, print_save_path=False, dpi=40,
    )
    palette = cmap if isinstance(cmap, ListedColormap) else matplotlib.colormaps[cmap or "viridis"]
    bars = plt.gcf().axes[0].containers[0]
    np.testing.assert_allclose([bar.get_facecolor() for bar in bars], palette([0, 0.5, 1]))
    assert list(bars.datavalues) == [1, 2, 3]
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_grouped_bars_preserve_series_colors(tmp_path):
    target = tmp_path / "grouped.png"
    bar_plot(
        ["a", "b"], [[1, 3], [2, 4]], cmap="Blues",
        series_labels=["first", "second"], gradient_start=0.2, gradient_end=0.8,
        sort_mode=None, save_path=target, print_save_path=False, dpi=40,
    )
    containers = plt.gcf().axes[0].containers
    colors = matplotlib.colormaps["Blues"]([0.2, 0.8])
    assert len(containers) == 2
    for bars, color in zip(containers, colors):
        np.testing.assert_allclose([bar.get_facecolor() for bar in bars], [color, color])
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("gradient", [False, True])
def test_line_plot_preserves_colormap_samples(gradient, tmp_path):
    target = tmp_path / "lines.png"
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
    [(fig, ax)] = line_plot(
        df, x="x", y="y", cmap_names=["viridis"], gradient=gradient,
        gradient_start=0.2, gradient_end=0.8, gradient_mode="linear",
        show=False, save_path=target, print_save_path=False, dpi=40,
    )
    fig.canvas.draw()
    palette = matplotlib.colormaps["viridis"]
    if gradient:
        np.testing.assert_allclose(ax.collections[0].get_colors(), palette([0.2, 0.8]))
    else:
        np.testing.assert_allclose(ax.lines[0].get_color(), palette(0.6))
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 1, 2])
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_box_plot_preserves_colors_labels_and_summary(tmp_path):
    target = tmp_path / "boxes.png"
    result = box_plot(
        {"first group": [1, 2, 3], "second group": [4, 5, 6]},
        cmap="viridis", wrap_width=7, print_summary=False,
        save_path=target, print_save_path=False, dpi=40,
    )
    ax = result["ax"]
    expected = matplotlib.colormaps["viridis"]([0.25, 0.9])
    expected[:, 3] = 0.9
    np.testing.assert_allclose([box.get_facecolor() for box in ax.patches], expected)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["first\ngroup", "second\ngroup"]
    np.testing.assert_array_equal(ax.get_xticks(), [1, 2])
    assert result["summary"]["median"].tolist() == [2, 5]
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_regression_plot_preserves_binned_scatter_colors(tmp_path):
    target = tmp_path / "regression.png"
    xs = np.arange(24, dtype=float)
    df = pd.DataFrame({"x": xs, "y": 0.5 * xs + np.sin(xs)})
    regression_plot(
        df, x="x", y="y", bins=4, cmap="viridis", print_summary=False,
        latex_options=False, save_path=target, print_save_path=False, dpi=40,
    )
    scatter = plt.gcf().axes[0].collections[-1]
    np.testing.assert_allclose(
        scatter.get_facecolors(), matplotlib.colormaps["viridis"](np.linspace(0, 1, 4))
    )
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
