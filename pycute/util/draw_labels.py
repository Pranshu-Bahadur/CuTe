# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Labelling functors shared by the CuTe Layout visualizers (draw_svg / draw_latex)

A `cell_label` functor gives the text of a cell, and an `axis_label` functor the
text of a row or column index in the margins. Each drawer takes both:

  -- `cell_label(val) -> str` for `draw_svg` / `draw_latex`, keyed on the
     cell's value -- the element of a `Tensor`, the offset of a `Layout` -- and
     for `draw_svg_inv` / `draw_latex_inv`, keyed on the index of the layout's
     domain sent to the cell; defaulting to `str`
  -- `cell_label(tid, vid) -> str` for `draw_svg_tv` / `draw_latex_tv`, keyed
     on the `(tid, vid)` pair that lands on the cell and defaulting to
     `tv_label`
  -- `axis_label(axis, i) -> str` for every drawer, naming row `i` when `axis`
     is `0` and column `i` when it is `1`, and defaulting to `index_label`

A label with several lines, separated by `\\n`, is drawn one line under the
other, and an empty label writes nothing: its cell is left blank, or its row or
column unlabelled.
"""

__all__ = ["tv_label", "index_label"]


def tv_label(tid, vid):
  """`T{tid}` over `V{vid}`: the default cell label of the thread-value drawers."""
  return f"T{tid}\nV{vid}"


def index_label(axis, i):
  """Row or column `i` by its index: the default axis label of every drawer."""
  return str(i)
