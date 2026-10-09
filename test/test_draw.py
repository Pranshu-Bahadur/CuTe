# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.util.draw_svg and pycute.util.draw_latex

These tests are also worked examples for docs/07_visualization.md.
"""

import ctypes
import re

import pytest

from pycute import *
from pycute.util import (bank_color_8x, draw_latex, draw_latex_inv, draw_latex_tv, draw_svg,
                         draw_svg_inv, draw_svg_tv, index_grey_8x, thread_color_8x, tv_label)

# A one-line cell label is a 10px Arial <text>, each line of a two-line one an
# 8px Arial <text>; the margin indices are 8px monospace.
SVG_CELL = re.compile(r'font-size="10px"[^>]*>([^<]*)<')
SVG_LINE = re.compile(r'font-family="Arial" font-size="8px"[^>]*>([^<]*)<')
SVG_AXIS = re.compile(r'font-family="monospace"[^>]*>([^<]*)<')
SVG_TEXT = re.compile(r'font-family="Arial"[^>]*x="(\d+)" y="(\d+)">([^<]*)<')
SVG_FILL = re.compile(r'fill="(rgb\([^"]*\))"')
TIKZ_CELL = re.compile(r"at \(\d+,\d+\) \{([^}]*)\}")
TIKZ_NODE = re.compile(r"^\\node\[fill=[^\]]*\] at \((\d+),(\d+)\) \{(.*)\};$", re.M)
TIKZ_AXIS = re.compile(r"\\texttt\{([^}]*)\}")
TIKZ_FILL = re.compile(r"fill=\{(rgb[^}]*)\}")


def svg_cells(text, cell_width=20):
  """`{(row, col): [line, ...]}` of the cell labels an SVG drawer wrote, which
  leaves a one-cell margin of 20 pixels for the indices."""
  cells = {}
  for x, y, line in SVG_TEXT.findall(text):
    cells.setdefault(((int(y) - 20) // 20, (int(x) - 20) // cell_width), []).append(line)
  return cells


def tikz_cells(text):
  """`{(row, col): label}` of the cell nodes a LaTeX drawer wrote."""
  return {(int(i), int(j)): label for i, j, label in TIKZ_NODE.findall(text)}


def _letters(layout) -> Tensor:
  """A tensor of letters, one per element of `layout`'s codomain."""
  T = make_tensor(layout, dtype=ctypes.c_wchar)
  for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: int(coshape(layout))]):
    T.accessor[i] = ch
  return T


@pytest.fixture(autouse=True)
def _require_svgwrite():
  pytest.importorskip("svgwrite")


class TestDrawSvg:

  def test_layout_labels_offsets(self, tmp_path, capsys):
    """A `Layout` is drawn as the offsets it produces."""
    out = tmp_path / "layout.svg"
    draw_svg(Layout((2, 4), (4, 1)), filename=str(out))
    assert "Saved as" in capsys.readouterr().out
    assert SVG_CELL.findall(out.read_text()) == ["0", "1", "2", "3", "4", "5", "6", "7"]

  def test_tensor_labels_elements(self, tmp_path):
    """A `Tensor` is drawn as the elements it holds, in layout order."""
    out = tmp_path / "tensor.svg"
    draw_svg(_letters(Layout((2, 4), (4, 1))), filename=str(out))
    assert SVG_CELL.findall(out.read_text()) == list("ABCDEFGH")

  def test_cells_are_white_by_default(self, tmp_path):
    out = tmp_path / "white.svg"
    draw_svg(Layout((2, 4), (4, 1)), filename=str(out))
    assert set(SVG_FILL.findall(out.read_text())) == {"rgb(255,255,255)"}

  def test_colored_by_offset_not_by_value(self, tmp_path):
    """A tensor and its layout shade identically: the color keys off the offset."""
    layout = Layout((2, 4), (4, 1))
    as_layout, as_tensor = tmp_path / "l.svg", tmp_path / "t.svg"
    draw_svg(layout, filename=str(as_layout), color=index_grey_8x)
    draw_svg(_letters(layout), filename=str(as_tensor), color=index_grey_8x)

    fills = SVG_FILL.findall(as_layout.read_text())
    assert len(set(fills)) > 1                     # actually shaded
    assert fills == SVG_FILL.findall(as_tensor.read_text())

  def test_f2_strided_layout_colors(self, tmp_path):
    """`F2` (swizzle) offsets reach the color functor as integers."""
    out = tmp_path / "swizzled.svg"
    draw_svg(Layout((8, 8), (F2(1), F2(9))), filename=str(out), color=bank_color_8x)
    assert len(set(SVG_FILL.findall(out.read_text()))) == 8

  def test_rank_1_drawn_as_single_row(self, tmp_path):
    """Rank-1 inputs are promoted to one row, as a tensor or a layout."""
    out = tmp_path / "rank1.svg"
    draw_svg(_letters(Layout(4, 1)), filename=str(out))
    assert SVG_CELL.findall(out.read_text()) == list("ABCD")

    draw_svg(Layout(4, 1), filename=str(out))
    assert SVG_CELL.findall(out.read_text()) == ["0", "1", "2", "3"]

  def test_rank_3_rejected(self, tmp_path):
    with pytest.raises(ValueError):
      draw_svg(Layout((2, 3, 4)), filename=str(tmp_path / "bad.svg"))

  def test_cell_width_fits_coordinate_labels(self, tmp_path):
    """`cell_width` widens the cells and leaves their height alone, so an
    `identity_tensor` can be labelled by its coordinates."""
    out = tmp_path / "coords.svg"
    draw_svg(identity_tensor((2, 3)), filename=str(out), cell_width=32,
             cell_label=lambda crd: "({},{})".format(*as_tuple(crd)))
    text = out.read_text()
    assert SVG_CELL.findall(text) == ["(0,0)", "(0,1)", "(0,2)", "(1,0)", "(1,1)", "(1,2)"]
    assert 'width="116"' in text and 'height="60"' in text   # 20 + 3*32 by 20 + 2*20

  def test_cell_label_and_axis_label_replace_the_defaults(self, tmp_path):
    """`cell_label` writes each cell's value, and `axis_label` each row and
    column index: rows first, then columns."""
    out = tmp_path / "labels.svg"
    draw_svg(Layout((2, 3), (3, 1)), filename=str(out),
             cell_label=lambda value: f"v{value}",
             axis_label=lambda axis, i: f"r{i}" if axis == 0 else "abc"[i])
    text = out.read_text()
    assert SVG_CELL.findall(text) == ["v0", "v1", "v2", "v3", "v4", "v5"]
    assert SVG_AXIS.findall(text) == ["r0", "r1", "a", "b", "c"]

  def test_an_empty_label_draws_nothing(self, tmp_path):
    """An empty label leaves its cell blank, or its row or column unlabelled."""
    out = tmp_path / "blanks.svg"
    draw_svg(Layout((2, 3), (3, 1)), filename=str(out),
             cell_label=lambda value: "" if value % 2 else str(value),
             axis_label=lambda axis, i: str(i) if axis == 0 else "")
    text = out.read_text()
    assert SVG_CELL.findall(text) == ["0", "2", "4"]
    assert SVG_AXIS.findall(text) == ["0", "1"]

  def test_a_label_of_several_lines_is_stacked(self, tmp_path):
    """Two lines share the cell, so each is drawn at the 8px that fits its half."""
    out = tmp_path / "lines.svg"
    draw_svg(Layout((1, 2), (0, 1)), filename=str(out),
             cell_label=lambda value: f"{value}\nb{value}")
    assert SVG_LINE.findall(out.read_text()) == ["0", "b0", "1", "b1"]


class TestDrawSvgInv:

  def test_each_value_is_an_index_of_the_domain(self, tmp_path):
    """`val` runs over the domain's indices, and is drawn where the layout sends it."""
    out = tmp_path / "inv.svg"
    seen = []
    draw_svg_inv(Layout((2, 2), (E(1), E(0))), filename=str(out),
                 cell_label=lambda val: seen.append(val) or val)
    assert seen == [0, 1, 2, 3]
    assert svg_cells(out.read_text()) == {(0, 0): ["0"], (0, 1): ["1"], (1, 0): ["2"], (1, 1): ["3"]}

  def test_the_inverse_of_a_layout_draws_the_layout(self, tmp_path):
    """crd -> val and its inverse val -> crd label every coordinate alike."""
    fwd, inv = tmp_path / "fwd.svg", tmp_path / "inv.svg"
    for L in (Layout((2, 3), (3, 1)), Layout(((2, 2), (2, 4)), ((1, 4), (2, 8)))):
      draw_svg(L, filename=str(fwd), color=bank_color_8x)
      draw_svg_inv(right_inverse(L), product_each(shape(L)), filename=str(inv), color=bank_color_8x)
      assert svg_cells(inv.read_text()) == svg_cells(fwd.read_text())
      assert sorted(SVG_FILL.findall(inv.read_text())) == sorted(SVG_FILL.findall(fwd.read_text()))

  def test_the_least_value_is_drawn_where_several_land(self, tmp_path):
    out = tmp_path / "window.svg"
    draw_svg_inv(Layout((3, 2), (E(1), E(1))), filename=str(out))
    assert svg_cells(out.read_text()) == {(0, 0): ["0"], (0, 1): ["1"], (0, 2): ["2"], (0, 3): ["5"]}

  def test_a_domain_of_any_rank(self, tmp_path):
    out = tmp_path / "rank3.svg"
    draw_svg_inv(Layout((2, 2, 2), (E(0), E(1), 2*E(1))), filename=str(out))
    assert len(svg_cells(out.read_text())) == 8

  def test_the_tile_is_the_grid(self, tmp_path):
    """A layout drawn into a larger tile leaves the rest of the tile white."""
    out = tmp_path / "partial.svg"
    draw_svg_inv(Layout((2, 2), (E(0), E(1))), (4, 4), filename=str(out))
    text = out.read_text()
    assert 'width="100"' in text and 'height="100"' in text   # 20 + 4*20 each way
    assert svg_cells(text) == {(0, 0): ["0"], (1, 0): ["1"], (0, 1): ["2"], (1, 1): ["3"]}

  def test_a_codomain_on_one_mode_is_drawn_in_the_tile(self, tmp_path):
    """Coordinates on mode 0 alone, and offsets that stay in the tile's first
    column, land in column 0."""
    out = tmp_path / "column.svg"
    for L in (Layout((2, 2), (E(0), 4*E(0))), Layout((2, 2), (1, 4))):
      draw_svg_inv(L, (6, 2), filename=str(out))
      assert svg_cells(out.read_text()) == {(0, 0): ["0"], (1, 0): ["1"], (4, 0): ["2"], (5, 0): ["3"]}

  def test_a_linear_offset_needs_a_rank_2_tile(self, tmp_path):
    """The tile defaults to the coshape, which a linear offset has only as a scalar."""
    for tile in (None, ((2, 2), (2, 1))):
      with pytest.raises(ValueError, match="rank-2 MN Tile"):
        draw_svg_inv(Layout((4, 2), (1, 4)), tile, filename=str(tmp_path / "bad.svg"))


class TestDrawSvgTv:

  def test_cells_show_thread_over_value_by_default(self, tmp_path):
    out = tmp_path / "tv.svg"
    draw_svg_tv(Layout((2, 2), (E(0), E(1))), filename=str(out))
    assert svg_cells(out.read_text()) == {(0, 0): ["T0", "V0"], (0, 1): ["T0", "V1"],
                                          (1, 0): ["T1", "V0"], (1, 1): ["T1", "V1"]}

  def test_the_inverse_view_by_thread_and_value(self, tmp_path):
    """`draw_svg_tv` is `draw_svg_inv`, with each index taken apart into `(tid, vid)`."""
    SM80 = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    S = product_each(shape(SM80))
    tv, inv = tmp_path / "tv.svg", tmp_path / "inv.svg"
    draw_svg_tv(SM80, (16, 8), filename=str(tv))
    draw_svg_inv(SM80, (16, 8), filename=str(inv),
                 color=lambda val: thread_color_8x(*idx2crd(val, S)),
                 cell_label=lambda val: tv_label(*idx2crd(val, S)))
    assert tv.read_text() == inv.read_text()

  def test_a_register_file_by_thread_and_value(self, tmp_path):
    """Rows named by thread and cells by value: the SIMT view of a TV layout."""
    out = tmp_path / "simt.svg"
    draw_svg_tv(Layout((2, 3), (E(0), E(1))), (2, 3), filename=str(out),
                cell_label=lambda tid, vid: f"V{vid}",
                axis_label=lambda axis, i: f"T{i}" if axis == 0 else "")
    text = out.read_text()
    assert svg_cells(text) == {(t, v): [f"V{v}"] for t in range(2) for v in range(3)}
    assert len(SVG_CELL.findall(text)) == 6
    assert SVG_AXIS.findall(text) == ["T0", "T1"]

  def test_a_rank_2_domain_is_required(self, tmp_path):
    with pytest.raises(ValueError, match="rank-2 TV"):
      draw_svg_tv(Layout((2, 2, 2), (E(0), E(1), 2*E(1))), filename=str(tmp_path / "bad.svg"))


class TestDrawLatex:

  def test_layout_labels_offsets(self, tmp_path):
    out = tmp_path / "layout.tex"
    draw_latex(Layout((2, 4), (4, 1)), filename=str(out), compile_pdf=False)
    assert TIKZ_CELL.findall(out.read_text()) == ["0", "1", "2", "3", "4", "5", "6", "7"]

  def test_tensor_labels_elements(self, tmp_path):
    out = tmp_path / "tensor.tex"
    draw_latex(_letters(Layout((2, 4), (4, 1))), filename=str(out), compile_pdf=False)
    text = out.read_text()
    assert TIKZ_CELL.findall(text) == list("ABCDEFGH")
    assert "% Layout: (2, 4):(4, 1)" in text   # the layout is recorded either way

  def test_cells_are_white_by_default(self, tmp_path):
    out = tmp_path / "white.tex"
    draw_latex(Layout((2, 4), (4, 1)), filename=str(out), compile_pdf=False)
    fills = set(TIKZ_FILL.findall(out.read_text()))
    assert fills == {"rgb,255:red,255;green,255;blue,255"}

  def test_rank_1_drawn_as_single_row(self, tmp_path):
    out = tmp_path / "rank1.tex"
    draw_latex(_letters(Layout(4, 1)), filename=str(out), compile_pdf=False)
    assert TIKZ_CELL.findall(out.read_text()) == list("ABCD")

  def test_rank_3_rejected(self, tmp_path):
    with pytest.raises(ValueError):
      draw_latex(Layout((2, 3, 4)), filename=str(tmp_path / "bad.tex"), compile_pdf=False)

  def test_cell_label_and_axis_label_replace_the_defaults(self, tmp_path):
    out = tmp_path / "labels.tex"
    draw_latex(Layout((2, 3), (3, 1)), filename=str(out), compile_pdf=False,
               cell_label=lambda value: f"v{value}",
               axis_label=lambda axis, i: f"r{i}" if axis == 0 else "")
    text = out.read_text()
    assert TIKZ_CELL.findall(text) == ["v0", "v1", "v2", "v3", "v4", "v5"]
    assert TIKZ_AXIS.findall(text) == ["r0", "r1"]

  def test_a_label_of_several_lines_is_stacked(self, tmp_path):
    """The default thread-value label is two lines, stacked as in `cute::print_latex`."""
    out = tmp_path / "tv.tex"
    draw_latex_tv(Layout((2, 2), (E(0), E(1))), filename=str(out), compile_pdf=False)
    assert "\\shortstack{T0 \\\\ V0}" in out.read_text()

    draw_latex_tv(Layout((2, 2), (E(0), E(1))), filename=str(out), compile_pdf=False,
                  cell_label=lambda tid, vid: f"V{vid}")
    assert tikz_cells(out.read_text()) == {(0, 0): "V0", (0, 1): "V1", (1, 0): "V0", (1, 1): "V1"}

  def test_the_inverse_of_a_layout_draws_the_layout(self, tmp_path):
    L = Layout((2, 3), (3, 1))
    fwd, inv = tmp_path / "fwd.tex", tmp_path / "inv.tex"
    draw_latex(L, filename=str(fwd), compile_pdf=False, color=bank_color_8x)
    draw_latex_inv(right_inverse(L), shape(L), filename=str(inv), compile_pdf=False,
                   color=bank_color_8x)
    assert tikz_cells(inv.read_text()) == tikz_cells(fwd.read_text())
    assert sorted(TIKZ_FILL.findall(inv.read_text())) == sorted(TIKZ_FILL.findall(fwd.read_text()))

  def test_tv_is_the_inverse_view_by_thread_and_value(self, tmp_path):
    SM80 = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    S = product_each(shape(SM80))
    tv, inv = tmp_path / "tv.tex", tmp_path / "inv.tex"
    draw_latex_tv(SM80, (16, 8), filename=str(tv), compile_pdf=False)
    draw_latex_inv(SM80, (16, 8), filename=str(inv), compile_pdf=False,
                   color=lambda val: thread_color_8x(*idx2crd(val, S)),
                   cell_label=lambda val: tv_label(*idx2crd(val, S)))
    assert tv.read_text() == inv.read_text()

  def test_an_empty_cell_label_keeps_the_fill(self, tmp_path):
    out = tmp_path / "blank.tex"
    draw_latex(Layout((1, 2), (0, 1)), filename=str(out), compile_pdf=False,
               color=lambda idx: (255, 0, 0), cell_label=lambda value: "")
    text = out.read_text()
    assert TIKZ_CELL.findall(text) == ["", ""]
    assert TIKZ_FILL.findall(text) == ["rgb,255:red,255;green,0;blue,0"] * 2
