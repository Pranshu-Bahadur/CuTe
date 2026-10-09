# Visualization

PyCuTe ships several printing and visualization utilities that, together,
cover almost all of your debugging and explanation needs:

* `print(layout)` / `print(tensor)` — the canonical
  `(shape):(stride)` notation (built into every PyCuTe object).
* [`print_tensor`](../pycute/util/print_tensor.py) — render any rank-1
  through rank-4 layout or tensor as nested ASCII tables.
* [`print_table`](../pycute/util/print_table.py) — render a rank-2 layout
  as a single bordered grid (uses `tabulate`).
* [`draw_svg`](../pycute/util/draw_svg.py) — render a rank-2 layout as
  a colored SVG file: a grid of coordinates, each labelled with the value
  the layout sends it to.
* [`draw_svg_inv`](../pycute/util/draw_svg.py) — the inverse view: render a
  layout that sends values to the coordinates of a grid, each coordinate
  labelled with the value sent there.
* [`draw_svg_tv`](../pycute/util/draw_svg.py) — `draw_svg_inv` of a rank-2
  thread-value layout, with `T#` / `V#` annotations.
* [`draw_latex`](../pycute/util/draw_latex.py) — render a rank-2 layout as
  a TikZ/LaTeX document (and PDF); the analogue of `cute::print_latex`.
* [`draw_latex_inv`](../pycute/util/draw_latex.py) and
  [`draw_latex_tv`](../pycute/util/draw_latex.py) — the TikZ/LaTeX analogues
  of `draw_svg_inv` and `draw_svg_tv`.
* [`draw_colors`](../pycute/util/draw_colors.py) — a small catalog of
  coloring functors (each returns an `(r, g, b)` RGB-255 tuple) shared by
  the SVG and LaTeX drawers: `white` (the default for `draw_svg` /
  `draw_latex`), `thread_color_8x(tid, vid)` (the default for the thread-value
  drawers), `index_grey_8x(idx)`, and the `bank_color_*` palettes. See
  [Coloring functors](#coloring-functors).
* [`draw_labels`](../pycute/util/draw_labels.py) — the labelling functors
  (each returns a string) of the same drawers: `tv_label(tid, vid)` (the
  `T#` / `V#` cell label of the thread-value drawers) and
  `index_label(axis, i)` (the row and column indices of every drawer). See
  [Labelling functors](#labelling-functors).

Every utility lives in [`pycute.util`](../pycute/util/), which is a
separate import from the core algebra:

```python
>>> from pycute import *
>>> from pycute.util import print_tensor, print_table
>>> from pycute.util import draw_svg, draw_svg_inv, draw_svg_tv
>>> from pycute.util import draw_latex, draw_latex_inv, draw_latex_tv
>>> from pycute.util import white, thread_color_8x          # default color functors
>>> from pycute.util import index_grey_8x, bank_color_32x   # ...and some alternatives
>>> from pycute.util import tv_label, index_label           # default label functors
```

The split is intentional: the core algebra has no third-party dependencies.
`print_table` requires [`tabulate`](https://pypi.org/project/tabulate/) and
the SVG helpers (`draw_svg`, `draw_svg_inv`, `draw_svg_tv`) require
[`svgwrite`](https://pypi.org/project/svgwrite/). The LaTeX helpers
(`draw_latex`, `draw_latex_inv`, `draw_latex_tv`) need no Python packages at all —
only a LaTeX installation (e.g. TeX Live providing `pdflatex`) for the
optional PDF step.

## `print` and `repr`

Every PyCuTe `Layout`, `Tensor`, `ArithTuple`, `F2`, and `Swizzle`
defines `__str__` and `__repr__`:

```python
>>> from pycute import *
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> print(A)
(3, (2, 4)):(2, (1, 6))
>>> repr(A)
'Layout((3, (2, 4)), (2, (1, 6)))'

>>> T = make_tensor(Layout((4, 4), (4, 1)))
>>> print(T)
<...Array...> o (4, 4):(4, 1)        # accessor o layout
```

`__str__` is the human-readable shape-and-stride notation. `__repr__` is
the full constructor form so you can copy-paste it into an interpreter.

## `print_tensor`

`print_tensor(t, print_type=True)` is defined in
[`print_tensor.py`](../pycute/util/print_tensor.py). It walks a rank-1
through rank-4 layout or tensor and prints it as nested ASCII tables.

### Rank-1 (vector)

```python
>>> from pycute.util import print_tensor
>>> print_tensor(Layout(8, 1))
8:1
0     1     2     3     4     5     6     7
```

### Rank-2 (matrix)

```python
>>> print_tensor(Layout((4, 8), (1, 4)))
(4, 8):(1, 4)
0     4     8     12    16    20    24    28
1     5     9     13    17    21    25    29
2     6     10    14    18    22    26    30
3     7     11    15    19    23    27    31

>>> print_tensor(Layout((3, (2, 3))))
(3, (2, 3)):(1, (3, 6))
0     3     6     9     12    15
1     4     7     10    13    16
2     5     8     11    14    17
```

The header line is suppressed by passing `print_type=False`.

### Rank-3 and rank-4

Higher-rank tensors are printed as 2-D slabs, separated by axis labels.
For a rank-3 tensor, you see one slab per `k` (the first slab, `k = 0`,
is printed without a separator line):

```
[ ... rank-2 slab for k=0 ... ]
--------  k = 1  ---------
[ ... rank-2 slab for k=1 ... ]
```

For a rank-4 tensor, slabs are grouped by the outermost `p` and separated
by `=` lines:

```
[ ... rank-3 slab for p=0 ... ]
==========  p = 1  ===========
[ ... rank-3 slab for p=1 ... ]
```

This is the same convention as `cute::print_tensor` in C++ CuTe.

### `print_tensor` on a Layout

When called on a `Layout`, `print_tensor` constructs a tensor backed by an
`ImplicitAccessor` and prints offsets:

```python
>>> print_tensor(Layout((4, 8), (1, 4)))
(4, 8):(1, 4)
0     4     8     12    16    20    24    28
1     5     9     13    17    21    25    29
2     6     10    14    18    22    26    30
3     7     11    15    19    23    27    31
```

This is the analogue of `cute::print_layout` in C++ CuTe. It is the
default visualization in this documentation.

## `print_table`

`print_table(t, print_type=True)` renders a rank-2 layout or tensor as a
single bordered grid (one cell per coordinate, the offset inside it). Like
`print_tensor`, it accepts either a `Tensor` or a `Layout` (the latter is
rendered through an `ImplicitAccessor`) and prints the `Shape:Stride`
header first unless `print_type=False`. It uses
[`tabulate`](https://pypi.org/project/tabulate/) for the table formatting,
so the output is friendlier than `print_tensor` when you want
copy-pasteable Markdown or fixed-width text:

```python
>>> from pycute.util import print_table
>>> print_table(Layout((4, 8), (1, 4)))
(4, 8):(1, 4)
+---+---+----+----+----+----+----+----+
| 0 | 4 |  8 | 12 | 16 | 20 | 24 | 28 |
+---+---+----+----+----+----+----+----+
| 1 | 5 |  9 | 13 | 17 | 21 | 25 | 29 |
+---+---+----+----+----+----+----+----+
| 2 | 6 | 10 | 14 | 18 | 22 | 26 | 30 |
+---+---+----+----+----+----+----+----+
| 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
+---+---+----+----+----+----+----+----+
```

For non-rank-2 inputs, `print_table` falls back to a plain `print`.

## `draw_svg`

[`draw_svg(tensor, filename="layout.svg", color=white, cell_width=20, cell_label=str, axis_label=index_label)`](../pycute/util/draw_svg.py)
visualizes `crd -> val`: it produces an SVG file rendering a layout as a grid
of coordinates `crd`, each cell labeled with `cell_label(val)` for the value
`val` there, its offset. Row and column indices are labeled along the left
and top edges.
Cells are white by default; pass a `color` functor to shade them, and a
`cell_label` or `axis_label` functor to relabel them.

```python
>>> from pycute.util import draw_svg
>>> draw_svg(Layout((4, 8), (1, 4)))
Saved as layout.svg
>>> draw_svg(Layout(((2, 2), (4, 2)), ((1, 8), (2, 16))),
...          filename="mixed.svg")
Saved as mixed.svg
```

Like [`print_tensor`](#print_tensor), it accepts a `Tensor` as well as a
`Layout`. A `Tensor` labels each cell with the *element* stored there while
the color still keys off the offset, so one figure shows both the logical
contents and where they live:

```python
>>> T = make_tensor(Layout((4, 8), (1, 4)))
>>> draw_svg(T, filename="data.svg", color=index_grey_8x)
Saved as data.svg
```

This renders an SVG grid — useful for slides, papers, and quick visual
inspection. For LaTeX/TikZ output, use [`draw_latex`](#draw_latex) below.

The arguments:

* `tensor` — a rank-2 `Tensor` or `Layout`; `size[0]` rows and `size[1]`
  columns are drawn. A rank-1 input is drawn as a single row.
* `filename` — defaults to `"layout.svg"`.
* `color` — a coloring functor `color(idx) -> (r, g, b)` mapping a cell's
  offset to an RGB-255 tuple (each component in `[0, 255]`). Defaults to
  `white`; the catalog in [Coloring functors](#coloring-functors) covers
  greyscale and bank palettes. The functors coerce the offset with `int()`,
  so `F2`-strided (swizzled) layouts color correctly too.
* `cell_width` — the width of a cell in pixels, `20` by default. Widen it for
  labels longer than a few characters, such as coordinates.
* `cell_label` — a labelling functor `cell_label(value) -> str` writing a
  cell's value: its offset, or the element of a `Tensor`. Defaults to `str`.
* `axis_label` — a labelling functor `axis_label(axis, i) -> str` writing
  the index `i` of a row (`axis == 0`) or a column (`axis == 1`). Defaults to
  `index_label`, the index itself. See
  [Labelling functors](#labelling-functors) for both.

The cell height and fonts are fixed — index labels are monospace, cell
labels are Arial; the palette is controlled by `color`, the text by
`cell_label` and `axis_label`.

For example, to highlight even versus odd offsets:

```python
>>> draw_svg(Layout((4, 8), (1, 4)),
...          color=lambda idx: (175, 255, 175) if idx % 2 == 0 else (255, 175, 175))
```

A coordinate prints in basis form when only one axis is nonzero, `7@1` for
`(0, 7)`, so to label every cell of an `identity_tensor` alike, write its
coordinates with a `cell_label`:

```python
>>> draw_svg(identity_tensor((4, 8)), filename="coords.svg", cell_width=32,
...          cell_label=lambda crd: "({},{})".format(*as_tuple(crd)))
Saved as coords.svg
```

## `draw_svg_inv`

[`draw_svg_inv(layout, tile_mn=None, filename="invlayout.svg", color=white, cell_label=str, axis_label=index_label)`](../pycute/util/draw_svg.py)
visualizes the inverse, `val -> crd`: `composition(tile_mn, layout)` sends
each index `val` of the layout's domain to a coordinate `crd` of the tile
`tile_mn`, and that cell is labeled with `cell_label(val)` and colored by
`color(val)`. The composition folds a linear offset into the tile's `(m, n)`
coordinates, and leaves `(m, n)` coordinates — a layout with `E(0)` / `E(1)`
strides — as they are. Where several values land on one cell the least is
drawn, and a cell no value lands on is left white.

The two views are inverse to each other: drawn with `draw_svg_inv`, the
inverse of a compact layout `L` labels every coordinate with the offset
`draw_svg(L)` gives it. For a hierarchical `L`, fold it into the flat tile
`product_each(shape(L))`.

```python
>>> from pycute.util import draw_svg_inv
>>> L = Layout((4, 8), (8, 1))                                   # crd -> offset
>>> draw_svg(L, filename="layout.svg")
Saved as layout.svg
>>> draw_svg_inv(right_inverse(L), shape(L), filename="inverse.svg")   # offset -> crd
Saved as inverse.svg
```

The arguments:

* `layout` — a layout, of any rank, whose codomain is a linear offset or
  `(m, n)` coordinates.
* `tile_mn` — the `(M, N)` tile drawn. Defaults to `coshape(layout)`, the
  extent of `(m, n)` coordinates; a linear offset needs it given.
* `filename` — defaults to `"invlayout.svg"`.
* `color` — a coloring functor `color(val) -> (r, g, b)`, keyed on the index
  `val` as `draw_svg`'s is on the offset, so the same catalog applies.
  Defaults to `white`.
* `cell_label` — a labelling functor `cell_label(val) -> str`. Defaults to
  `str`.
* `axis_label` — as in `draw_svg`.

Errors raised:

* `Expected a rank-2 MN Tile` — `tile_mn` is not a rank-2 shape of integers.
  You also get this when you pass a linear-offset layout without `tile_mn`,
  since the default `coshape(layout)` is then a scalar.

## `draw_svg_tv`

[`draw_svg_tv(layout, tile_mn=None, filename="tvlayout.svg", color=thread_color_8x, cell_label=tv_label, axis_label=index_label)`](../pycute/util/draw_svg.py)
is `draw_svg_inv` for a **thread-value layout**: a rank-2 layout whose
domain is `(tid, vid)`. This is exactly the layout that comes out of
partitioning a tile across the threads of a warp/threadblock. Each index of
the domain is taken apart into its thread and value, so the functors take
`(tid, vid)`, and the layout is composed into the tile as in `draw_svg_inv`.

```python
>>> from pycute.util import draw_svg_tv

>>> # (a) codomain already in (m, n) coordinates:
>>> tv = Layout((8, 8), (E(0), E(1)))
>>> draw_svg_tv(tv, filename="identity_tv.svg")
Saved as identity_tv.svg

>>> # (b) codomain as linear offsets: pass the (M, N) tile shape and the
>>> # offsets are mapped into it automatically:
>>> SM80 = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))   # (tid, vid) -> offset
>>> draw_svg_tv(SM80, tile_mn=(16, 8), filename="sm80_tv.svg")
Saved as sm80_tv.svg
```

Each cell of the resulting SVG shows (by default) `T<tid>` / `V<vid>` for
the `(tid, vid)` pair that lands on that `(m, n)` position, with the cell
colored (by default) by `tid mod 8`. Where several pairs land on one cell,
the one of least index `tid + T*vid` in the domain is drawn, `T` the number
of threads. Empty positions (no thread covers them) are left white.

The arguments are those of `draw_svg_inv`, except:

* `layout` — a rank-2 layout with domain `(tid, vid)`.
* `filename` — defaults to `"tvlayout.svg"`.
* `color` — a coloring functor `color(tid, vid) -> (r, g, b)` mapping a
  `(tid, vid)` pair to an RGB-255 tuple (each component in `[0, 255]`).
  Defaults to `thread_color_8x` (coloring by `tid % 8`), importable from
  `pycute.util`. Because it receives both `tid` and `vid`, you can color by
  value index instead (e.g. `color=lambda tid, vid: (200, 200, 255) if vid %
  2 else (255, 200, 200)`).
* `cell_label` — a labelling functor `cell_label(tid, vid) -> str` writing
  the cell of a `(tid, vid)` pair. Defaults to `tv_label`, `T<tid>` over
  `V<vid>`.

Errors raised: those of `draw_svg_inv`, and

* `Expected a rank-2 TV Layout` — the layout's domain is not rank 2.

## `draw_latex`

[`draw_latex(tensor, filename="layout.tex", compile_pdf=True, color=white, cell_label=str, axis_label=index_label)`](../pycute/util/draw_latex.py)
is the LaTeX/PDF analogue of `draw_svg`. It writes a standalone
[TikZ](https://tikz.dev/) document rendering a rank-2 layout as a grid — cells
labeled by offset and (by default) unshaded, with row/column index labels —
and then compiles it to a cropped PDF with `pdflatex`. The output mirrors
`cute::print_latex` in C++ CuTe, so it is ideal for slides and papers.

```python
>>> from pycute.util import draw_latex
>>> draw_latex(Layout((4, 8), (1, 4)))
Saved as layout.tex
Saved as layout.pdf
```

The arguments:

* `tensor` — a rank-2 `Tensor` or `Layout`. As with `draw_svg`, a `Tensor`
  labels each cell with the element stored there rather than its offset, and
  a rank-1 input is drawn as a single row.
* `filename` — the `.tex` path; defaults to `"layout.tex"`. The PDF is
  written next to it with the same stem (`layout.pdf`).
* `compile_pdf` — when `True` (default), run `pdflatex` to produce the PDF
  and remove the `.aux`/`.log` byproducts. When `False`, only the `.tex`
  is written.
* `color` — a coloring functor `color(idx) -> (r, g, b)`, identical in
  contract to `draw_svg`'s `color` (RGB-255 tuple, components in
  `[0, 255]`). Defaults to `white`.
* `cell_label`, `axis_label` — labelling functors, identical in contract to
  `draw_svg`'s. A label of several lines is set with `\shortstack`.

Unlike `draw_svg`, `draw_latex` needs **no Python packages** — only a LaTeX
installation (e.g. TeX Live providing `pdflatex`) for the PDF step. If
`pdflatex` is not found on `PATH`, the `.tex` is still written and a note
is printed instead of raising.

## `draw_latex_inv`

[`draw_latex_inv(layout, tile_mn=None, filename="invlayout.tex", compile_pdf=True, color=white, cell_label=str, axis_label=index_label)`](../pycute/util/draw_latex.py)
is the LaTeX/PDF analogue of `draw_svg_inv`, visualizing `val -> crd`. It
accepts the same layouts and tile, takes the same functors, and raises the
same errors. The `filename` and `compile_pdf` arguments behave as in
`draw_latex`.

```python
>>> from pycute.util import draw_latex_inv
>>> L = Layout((4, 8), (8, 1))
>>> draw_latex_inv(right_inverse(L), shape(L))
Saved as invlayout.tex
Saved as invlayout.pdf
```

## `draw_latex_tv`

[`draw_latex_tv(layout, tile_mn=None, filename="tvlayout.tex", compile_pdf=True, color=thread_color_8x, cell_label=tv_label, axis_label=index_label)`](../pycute/util/draw_latex.py)
is the LaTeX/PDF analogue of `draw_svg_tv`: `draw_latex_inv` for a
thread-value layout, with the functors taking `(tid, vid)`. Each cell shows
(by default) `T<tid>` / `V<vid>` for the `(tid, vid)` landing there, colored
by the `color` functor (`color(tid, vid) -> (r, g, b)`, defaulting to
`thread_color_8x`). The `tile_mn`, `color`, `cell_label`, `axis_label`,
`filename`, and `compile_pdf` arguments behave as in `draw_svg_tv` /
`draw_latex`, and the errors are those of `draw_svg_tv`.

```python
>>> from pycute.util import draw_latex_tv
>>> SM80 = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))   # (tid, vid) -> offset
>>> draw_latex_tv(SM80, tile_mn=(16, 8))
Saved as tvlayout.tex
Saved as tvlayout.pdf
```

## Coloring functors

Every drawer takes a `color` functor that maps a cell key to an `(r, g, b)`
tuple (each component in `[0, 255]`). There are two key shapes, matching the
two families of drawers:

* **offset** functors — `color(idx) -> (r, g, b)` — for `draw_svg` /
  `draw_latex`, keyed on the cell's integer offset, and for `draw_svg_inv` /
  `draw_latex_inv`, keyed on the integer `val` drawn there.
* **thread-value** functors — `color(tid, vid) -> (r, g, b)` — for
  `draw_svg_tv` / `draw_latex_tv`, keyed on the `(tid, vid)` pair.

[`pycute.util.draw_colors`](../pycute/util/draw_colors.py) provides a small
catalog of ready-made functors:

| Functor | Signature | Colors by |
|---|---|---|
| `white` *(default for the other drawers)* | either (`*args`) | constant white |
| `index_grey_8x` | `color(idx)` | greyscale, `idx % 8` |
| `bank_color_8x` | `color(idx)` | shared-memory bank, `idx % 8` (light spectrum) |
| `bank_color_16x` | `color(idx)` | shared-memory bank, `idx % 16` (light spectrum) |
| `bank_color_32x` | `color(idx)` | shared-memory bank, `idx % 32` (light spectrum) |
| `thread_color_8x` *(default for the TV drawers)* | `color(tid, vid)` | thread, `tid % 8` |
| `value_color_8x` | `color(tid, vid)` | value index, `vid % 8` |
| `warp_color_8x` | `color(tid, vid)` | warp, `(tid // 32) % 8` |
| `constant(rgb)` | factory → either | constant `rgb` |

The three `bank_color_*` palettes are evenly-spaced subsamples of one shared
32-color light spectrum, so they stay mutually consistent (pick the modulus
that matches your bank count or how many distinct colors read best).

```python
>>> from pycute.util import (draw_svg, draw_svg_tv, bank_color_32x,
...                          value_color_8x, warp_color_8x, white, constant)

>>> # Spot shared-memory bank conflicts: cells sharing a bank share a color.
>>> draw_svg(Layout((8, 4), (4, 1)), color=bank_color_32x)
Saved as layout.svg

>>> # Color a thread-value layout by which warp owns each cell:
>>> tv = Layout((8, 8), (E(0), E(1)))
>>> draw_svg_tv(tv, color=warp_color_8x)
Saved as tvlayout.svg

>>> # ...or by value index instead of thread:
>>> draw_svg_tv(tv, color=value_color_8x)
Saved as tvlayout.svg

>>> # Flat coloring (e.g. for a clean figure), via white or a constant:
>>> draw_svg(Layout((4, 8), (1, 4)), color=white)
Saved as layout.svg
>>> draw_svg(Layout((4, 8), (1, 4)), color=constant((255, 225, 180)))
Saved as layout.svg
```

Anything callable works: to roll your own, just supply a function (or lambda)
of the matching signature that returns an `(r, g, b)` tuple, e.g.
`color=lambda idx: (255, 175, 175) if idx % 2 else (175, 255, 175)`.

## Labelling functors

Every drawer also takes two labelling functors, each returning the text to
write. A `"\n"` in the text starts a new line, so a cell can carry a label of
several lines, as the thread-value cells do by default, and an empty text
writes nothing.

* **cell** labels write each cell — `cell_label(val) -> str` for
  `draw_svg` / `draw_latex`, given the cell's offset or, for a `Tensor`, its
  element, and for `draw_svg_inv` / `draw_latex_inv`, given the index `val`
  that lands there; `cell_label(tid, vid) -> str` for `draw_svg_tv` /
  `draw_latex_tv`, given the `(tid, vid)` pair that lands there. An empty
  label leaves the cell blank, still colored.
* **axis** labels write the margins — `axis_label(axis, i) -> str` for every
  drawer, given row `i` (`axis == 0`, down the left edge) or column `i`
  (`axis == 1`, along the top). An empty label leaves that row or column
  unlabelled.

[`pycute.util.draw_labels`](../pycute/util/draw_labels.py) provides the
defaults:

| Functor | Signature | Writes |
|---|---|---|
| `str` *(default for the other drawers)* | `cell_label(val)` | the value |
| `tv_label` *(default for the TV drawers)* | `cell_label(tid, vid)` | `T<tid>` over `V<vid>` |
| `index_label` *(default for every drawer)* | `axis_label(axis, i)` | the index `i` |

```python
>>> from pycute.util import draw_svg, draw_svg_tv

>>> # Rows and columns named by mode, m0..m3 and n0..n7:
>>> draw_svg(Layout((4, 8), (1, 4)), axis_label=lambda axis, i: f"{'mn'[axis]}{i}")
Saved as layout.svg

>>> # A thread-value layout as a register file: a row per thread, a value per cell.
>>> tv = Layout((4, 6), (E(0), E(1)))
>>> draw_svg_tv(tv, cell_label=lambda tid, vid: f"V{vid}",
...             axis_label=lambda axis, i: f"T{i}" if axis == 0 else "")
Saved as tvlayout.svg
```

## Putting it together: a worked example

A complete debugging session using the visualization toolkit:

```python
>>> from pycute import *
>>> from pycute.util import *

>>> # Build a 4x8 identity tile (every cell is its own (m, n) coordinate):
>>> tile = Layout((4, 8), (E(0), E(1)))

>>> # Print as ASCII offsets:
>>> print_tensor(Layout((4, 8), (1, 4)))
(4, 8):(1, 4)
0     4     8     12    16    20    24    28
1     5     9     13    17    21    25    29
2     6     10    14    18    22    26    30
3     7     11    15    19    23    27    31

>>> # Save SVG visualizations:
>>> draw_svg(Layout((4, 8), (1, 4)), "data_layout.svg")
Saved as data_layout.svg
>>> draw_svg_tv(tile, filename="tile_coords.svg")
Saved as tile_coords.svg

>>> # ...or as LaTeX/PDF (same picture, vector output for papers):
>>> draw_latex(Layout((4, 8), (1, 4)), "data_layout.tex")
Saved as data_layout.tex
Saved as data_layout.pdf
```

## Source and tests

* Source: [`pycute/util/print_tensor.py`](../pycute/util/print_tensor.py),
  [`pycute/util/print_table.py`](../pycute/util/print_table.py),
  [`pycute/util/draw_svg.py`](../pycute/util/draw_svg.py),
  [`pycute/util/draw_latex.py`](../pycute/util/draw_latex.py),
  [`pycute/util/draw_colors.py`](../pycute/util/draw_colors.py),
  [`pycute/util/draw_labels.py`](../pycute/util/draw_labels.py).
* Tests: [`test/test_draw.py`](../test/test_draw.py) asserts on what the
  drawers write — labels, fills and sizes, and the inverse and thread-value
  views. The printers are not unit-tested, but you can drive `print_tensor`
  from any layout-algebra test as a quick visual sanity check.

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
