# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for LaTeX/PDF generation of CuTe Layouts
"""

from typing import Union

from pycute import *

from .draw_colors import thread_color_8x, white
from .draw_labels import index_label, tv_label


_latex_header = (
  "\\documentclass{standalone}\n"
  "\\usepackage{tikz}\n\n"
  "\\begin{document}\n"
  "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},"
  "every node/.style={minimum size=1cm, outer sep=0pt}]\n\n"
)

_latex_footer = (
  "\\end{tikzpicture}\n"
  "\\end{document}\n"
)


def _tikz_rgb(rgb):
  r, g, b = rgb
  return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def _draw_label(text, x, y, font_size="", font_family="", fill=None):
  # A node at `(x, y)` writing `text`, its lines stacked one under the other,
  # on the cell's `fill` if it has one; a node with neither is left out.
  lines = str(text).split("\n")
  text = lines[0] if len(lines) == 1 else "\\shortstack{" + " \\\\ ".join(lines) + "}"
  if not text and fill is None:
    return ""
  if font_family:
    text = f"{font_family}{{{text}}}"
  if font_size:
    text = f"{font_size}{{{text}}}"
  options = "" if fill is None else f"[fill={_tikz_rgb(fill)}]"
  return f"\\node{options} at ({x},{y}) {{{text}}};\n"


def _write_and_compile(body, filename, compile_pdf):
  import os
  import shutil
  import subprocess

  doc = _latex_header + body + _latex_footer
  with open(filename, "w") as f:
    f.write(doc)
  print(f"Saved as {filename}")

  if not compile_pdf:
    return

  pdflatex = shutil.which("pdflatex")
  if pdflatex is None:
    print(f"  (pdflatex not found on PATH; wrote .tex only. Install a LaTeX "
          f"distribution, or compile manually with `pdflatex {filename}`.)")
    return

  stem = os.path.splitext(filename)[0]
  out_dir = os.path.dirname(os.path.abspath(filename))
  try:
    subprocess.run(
      [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
       "-output-directory", out_dir, filename],
      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
  except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    print(f"  (pdflatex failed; wrote .tex only. See {stem}.log for details.)")
    return

  for ext in (".aux", ".log"):
    aux = stem + ext
    if os.path.exists(aux):
      os.remove(aux)
  print(f"Saved as {stem}.pdf")


def draw_latex(tensor : Union[Tensor, LayoutBase], filename="layout.tex", compile_pdf=True, color=white,
               cell_label=str, axis_label=index_label):
  """
  Visualize crd -> val over a grid of coordinates, crd, labelled with cell_label(val).

  The LaTeX/PDF analogue of `draw_svg`, mirroring `cute::print_latex`. Writes a
  standalone TikZ document for a rank-2 `Tensor` or `Layout` (rank-1 becomes a
  single row) and, by default, compiles it to a cropped PDF with `pdflatex`. A
  missing or failing `pdflatex` leaves the `.tex` in place rather than
  raising. `color`, `cell_label` and `axis_label` have the same contracts as
  in `draw_svg`.

  Pre-conditions:
    rank(tensor) is 1 or 2; otherwise a ValueError is raised
  """
  if isinstance(tensor, LayoutBase):
    tensor = Tensor(ImplicitAccessor(0), tensor)
  if rank(tensor) == 1:
    tensor = Tensor(tensor.accessor, make_layout([Layout(1,0), tensor.layout]))
  if rank(tensor) != 2:
    raise ValueError(f"Expected a rank-2 Layout, got {tensor.layout}")

  M, N = size[0](tensor), size[1](tensor)

  body = f"% Layout: {tensor.layout}\n"
  for i in range(M):
    for j in range(N):
      # Label by the value, color by the offset
      value  = tensor[i, j]
      offset = tensor.layout(i, j)
      body += _draw_label(cell_label(value), i, j, fill=color(offset))
  body += f"\n\\draw[color=black,thick,shift={{(-0.5,-0.5)}}] (0,0) grid ({M},{N});\n\n"

  # Row indices down the left margin, column indices across the top
  for i in range(M):
    body += _draw_label(axis_label(0, i), i, -1, "\\Large", "\\texttt")
  for j in range(N):
    body += _draw_label(axis_label(1, j), -1, j, "\\Large", "\\texttt")

  _write_and_compile(body, filename, compile_pdf)


def draw_latex_inv(layout : LayoutBase, tile_mn=None, filename="invlayout.tex", compile_pdf=True, color=white,
                   cell_label=str, axis_label=index_label):
  """
  Visualize val -> crd over a grid of coordinates, crd, labelled with cell_label(val).

  The LaTeX/PDF analogue of `draw_svg_inv`: the same layouts, tile, functors
  and errors, rendered as TikZ and compiled to PDF by default.

  Pre-conditions:
    tile_mn is a rank-2 shape; otherwise a ValueError is raised
  """
  tile_mn = coshape(layout) if tile_mn is None else tile_mn
  if not congruent(tile_mn, (0,0)):
    raise ValueError(f"Expected a rank-2 MN Tile")
  layout = composition(tile_mn, layout)

  M, N = tile_mn
  filled = [[False for n in range(N)] for m in range(M)]

  # Zero-extended, since a coordinate on mode 0 alone is sparse: `5@0` is (5, 0)
  body = f"% Layout: {layout}\n"
  for val in range(size(layout)):
    i, j = ArithTuple(0, 0) + layout(val)
    if filled[i][j]:
      continue
    filled[i][j] = True
    body += _draw_label(cell_label(val), i, j, fill=color(val))
  body += f"\n\\draw[color=black,thick,shift={{(-0.5,-0.5)}}] (0,0) grid ({M},{N});\n\n"

  # Row indices down the left margin, column indices across the top
  for i in range(M):
    body += _draw_label(axis_label(0, i), i, -1, "\\Large", "\\texttt")
  for j in range(N):
    body += _draw_label(axis_label(1, j), -1, j, "\\Large", "\\texttt")

  _write_and_compile(body, filename, compile_pdf)


def draw_latex_tv(layout : LayoutBase, tile_mn=None, filename="tvlayout.tex", compile_pdf=True, color=thread_color_8x,
                  cell_label=tv_label, axis_label=index_label):
  """
  Visualize a thread-value layout, (tid, vid) -> crd, labelled with cell_label(tid, vid).

  The LaTeX/PDF analogue of `draw_svg_tv`: `draw_latex_inv`, with each index of
  the domain taken apart into its thread `tid` and value `vid` for `color` and
  `cell_label`.

  Pre-conditions:
    rank(layout) == 2; otherwise a ValueError is raised
  """
  if rank(layout) != 2:
    raise ValueError(f"Expected a rank-2 TV Layout")

  S = product_each(shape(layout))
  return draw_latex_inv(layout, tile_mn, filename, compile_pdf,
                        lambda val: color(*idx2crd(val, S)),
                        lambda val: cell_label(*idx2crd(val, S)), axis_label)
