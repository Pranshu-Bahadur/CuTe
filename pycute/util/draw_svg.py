# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for SVG generation of CuTe Layouts
"""

from typing import Union

from pycute import *

from .draw_colors import thread_color_8x, white
from .draw_labels import index_label, tv_label


def _draw_label(dwg, text, x, y, cell_width, cell_height, font_size, font_family):
  # Each line of `text` is centered in its share of the cell, one under the
  # other, at `font_size` pixels or less: a line takes 1.25 times its font
  # size. An empty line draws nothing. `text-anchor=middle` + `dy=0.35em`
  # centers a line on its point in a renderer-portable way: it relies only on
  # baseline positioning (universally supported), not on `dominant-baseline`,
  # which several SVG renderers ignore on <text> elements.
  lines = str(text).split("\n")
  font_size = min(font_size, 4*cell_height // (5*len(lines)))
  for k, line in enumerate(lines):
    if line:
      dwg.add(dwg.text(
        line, insert=(x + cell_width//2, y + (2*k + 1)*cell_height//(2*len(lines))),
        dy=["0.35em"], text_anchor="middle", font_size=f"{font_size}px", font_family=font_family))


def draw_svg(tensor : Union[Tensor, LayoutBase], filename="layout.svg", color=white, cell_width=20,
             cell_label=str, axis_label=index_label):
  """
  Visualize crd -> val over a grid of coordinates, crd, labelled with cell_label(val).

  Saves a rank-2 `Tensor` or `Layout` as an SVG table with row/column labels:
  a `Layout` labels each cell with its offset, a `Tensor` with the element
  stored there. A rank-1 input is drawn as a single row. `color` is a functor
  `color(idx) -> (r, g, b)` keyed on the cell's offset; `draw_colors` holds a
  catalog of them. `cell_width` widens the cells, in pixels, for labels longer
  than a few characters, such as coordinates. `cell_label(val) -> str` writes
  a cell's value, and `axis_label(axis, i) -> str` names row `i` (axis `0`) or
  column `i` (axis `1`); `draw_labels` holds the defaults.

  Requires the optional `svgwrite` package (`pip install pycute[viz]`).

  Pre-conditions:
    rank(tensor) is 1 or 2; otherwise a ValueError is raised
  """
  try:
    import svgwrite
  except ImportError as e:
    raise ImportError(
      "draw_svg requires the optional 'svgwrite' package. "
      "Install it with `pip install svgwrite` or `pip install pycute[viz]`."
    ) from e

  if isinstance(tensor, LayoutBase):
    tensor = Tensor(ImplicitAccessor(0), tensor)
  if rank(tensor) == 1:
    tensor = Tensor(tensor.accessor, make_layout([Layout(1,0), tensor.layout]))
  if rank(tensor) != 2:
    raise ValueError(f"Expected a rank-2 Layout, got {tensor.layout}")

  # Cell size in pixels, with a one-cell margin for the index labels
  cell_size = 20
  margin = cell_size

  # Grid size
  M, N = size[0](tensor), size[1](tensor)

  # Create SVG canvas
  dwg = svgwrite.Drawing(filename, size=(margin + N*cell_width, margin + M*cell_size))

  # Draw grid cells
  for i in range(M):
    for j in range(N):
      # Label by the value, color by the offset
      value  = tensor[i, j]
      offset = tensor.layout(i, j)
      x = margin + j * cell_width
      y = margin + i * cell_size

      # Draw rectangle
      dwg.add(dwg.rect(
        insert=(x, y), size=(cell_width, cell_size),
        fill=svgwrite.rgb(*color(offset), mode='RGB'), stroke='black'))

      # Add label text
      _draw_label(dwg, cell_label(value), x, y, cell_width, cell_size, 10, "Arial")

  # Row indices down the left margin, column indices across the top
  for i in range(M):
    _draw_label(dwg, axis_label(0, i), 0, margin + i*cell_size, margin, cell_size, 8, "monospace")
  for j in range(N):
    _draw_label(dwg, axis_label(1, j), margin + j*cell_width, 0, cell_width, margin, 8, "monospace")

  dwg.save()
  print(f"Saved as {filename}")


def draw_svg_inv(layout : LayoutBase, tile_mn=None, filename="invlayout.svg", color=white,
                 cell_label=str, axis_label=index_label):
  """
  Visualize val -> crd over a grid of coordinates, crd, labelled with cell_label(val).

  The inverse view of `draw_svg`: `composition(tile_mn, layout)` sends each
  index `val` of the layout's domain to a coordinate `(m, n)` of the tile
  `tile_mn`, folding a linear offset into the tile and leaving a coordinate as
  it is, and `color(val) -> (r, g, b)` and `cell_label(val) -> str` color and
  label that cell. `tile_mn` defaults to `coshape(layout)`. Where several
  values land on one cell the least is drawn, and a cell none lands on is left
  white. `axis_label` is as in `draw_svg`.

  Requires the optional `svgwrite` package (`pip install pycute[viz]`).

  Pre-conditions:
    tile_mn is a rank-2 shape; otherwise a ValueError is raised
  """
  try:
    import svgwrite
  except ImportError as e:
    raise ImportError(
      "draw_svg_inv requires the optional 'svgwrite' package. "
      "Install it with `pip install svgwrite` or `pip install pycute[viz]`."
    ) from e

  tile_mn = coshape(layout) if tile_mn is None else tile_mn
  if not congruent(tile_mn, (0,0)):
    raise ValueError(f"Expected a rank-2 MN Tile")
  layout = composition(tile_mn, layout)

  # Cell size in pixels, with a one-cell margin for the index labels
  cell_size = 20
  margin = cell_size

  # Grid size
  M, N = tile_mn
  filled = [[False for n in range(N)] for m in range(M)]

  # Create SVG canvas
  dwg = svgwrite.Drawing(filename, size=(margin + N*cell_size, margin + M*cell_size))

  # Fill in grid (empty cells are white)
  for i in range(M):
    for j in range(N):
      dwg.add(dwg.rect(
        insert=(margin + j*cell_size, margin + i*cell_size), size=(cell_size, cell_size),
        fill='white', stroke='black'))

  # Draw each value at its coordinate, the least first. Zero-extended, since a
  # coordinate on mode 0 alone is sparse: `5@0` is (5, 0).
  for val in range(size(layout)):
    i, j = ArithTuple(0, 0) + layout(val)
    if filled[i][j]:
      continue
    filled[i][j] = True

    x = margin + j * cell_size
    y = margin + i * cell_size

    # Draw rectangle
    dwg.add(dwg.rect(
      insert=(x, y), size=(cell_size, cell_size),
      fill=svgwrite.rgb(*color(val), mode='RGB'), stroke='black'))

    # Add label text
    _draw_label(dwg, cell_label(val), x, y, cell_size, cell_size, 10, "Arial")

  # Row indices down the left margin, column indices across the top
  for i in range(M):
    _draw_label(dwg, axis_label(0, i), 0, margin + i*cell_size, margin, cell_size, 8, "monospace")
  for j in range(N):
    _draw_label(dwg, axis_label(1, j), margin + j*cell_size, 0, cell_size, margin, 8, "monospace")

  dwg.save()
  print(f"Saved as {filename}")


def draw_svg_tv(layout : LayoutBase, tile_mn=None, filename="tvlayout.svg", color=thread_color_8x,
                cell_label=tv_label, axis_label=index_label):
  """
  Visualize a thread-value layout, (tid, vid) -> crd, labelled with cell_label(tid, vid).

  `draw_svg_inv`, with each index of the domain taken apart into its thread
  `tid` and value `vid`: `color(tid, vid) -> (r, g, b)` colors a cell, by
  thread by default, and `cell_label(tid, vid) -> str` labels it, `T{tid}`
  over `V{vid}` by default. The other arguments are as in `draw_svg_inv`.

  Pre-conditions:
    rank(layout) == 2; otherwise a ValueError is raised
  """
  if rank(layout) != 2:
    raise ValueError(f"Expected a rank-2 TV Layout")

  S = product_each(shape(layout))
  return draw_svg_inv(layout, tile_mn, filename,
                      lambda val: color(*idx2crd(val, S)),
                      lambda val: cell_label(*idx2crd(val, S)), axis_label)
