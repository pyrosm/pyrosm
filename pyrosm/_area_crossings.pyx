# cython: language_level=3
"""Geometry of the lines across walking areas (see :mod:`pyrosm.walking_areas`): which
straight lines between points of an area lie inside it, the shortest paths over those
lines, which paths pass another entrance, and which pairs of entrances get a line.

Coordinates are longitude and latitude. Where rounding leaves an answer open (a line
that touches the outline, runs along it or passes a point within the tolerance), the
functions say so, and the caller asks shapely, so the results are those of
:func:`shapely.covers` and :func:`shapely.dwithin`. Index arrays are ``np.intp``.
"""
import numpy as np
cimport cython
from libc.math cimport INFINITY, fabs, sqrt

# The relative error bound of the orientation test without exact arithmetic (Shewchuk's
# ccwerrboundA).
cdef double _ORIENT_ERROR = 3.3306690738754716e-16

# A point this many degrees (about 2 mm) from the outline and outside the area lies outside
# the area widened by the walking areas' tolerance, also at a mitred corner.
cdef double _CLEAR = 2e-8

# The first step along a line, in degrees, from an end where it leaves the area to a
# point that may lie clearly outside.
cdef double _STEP = 4e-8

# Cells of the grid over an area hold the edges within this many degrees of them, more
# than rounding can move a point.
cdef double _GRID_MARGIN = 1e-9

cdef enum:
    _HIDDEN = 0
    _VISIBLE = 1
    _OPEN = 2

HIDDEN = _HIDDEN
VISIBLE = _VISIBLE
OPEN = _OPEN


cdef inline int _orientation(
    double ax, double ay, double bx, double by, double cx, double cy
) noexcept nogil:
    """1 when ``c`` lies to the left of the line from ``a`` to ``b``, -1 to its right,
    0 on it or too close to tell."""
    cdef double left = (ax - cx) * (by - cy)
    cdef double right = (ay - cy) * (bx - cx)
    cdef double det = left - right
    cdef double total
    if left > 0:
        if right <= 0:
            return (det > 0) - (det < 0)
        total = left + right
    elif left < 0:
        if right >= 0:
            return (det > 0) - (det < 0)
        total = -left - right
    else:
        return (det > 0) - (det < 0)
    if fabs(det) >= _ORIENT_ERROR * total:
        return (det > 0) - (det < 0)
    return 0


cdef inline double _squared_distance(
    double px, double py, double ax, double ay, double bx, double by
) noexcept nogil:
    """The squared distance from the point ``p`` to the segment from ``a`` to ``b``."""
    cdef double dx = bx - ax, dy = by - ay
    cdef double length = dx * dx + dy * dy
    cdef double t = 0.0
    if length > 0:
        t = ((px - ax) * dx + (py - ay) * dy) / length
        if t < 0:
            t = 0.0
        elif t > 1:
            t = 1.0
    dx = px - (ax + t * dx)
    dy = py - (ay + t * dy)
    return dx * dx + dy * dy


cdef inline Py_ssize_t _cell_index(
    double value, double start, double size, Py_ssize_t count
) noexcept nogil:
    """The index of the cell of ``size`` that holds ``value`` in a row of ``count``
    cells from ``start``, the nearest one where ``value`` lies outside the row."""
    cdef double at = (value - start) / size
    if at < 0:
        return 0
    if at >= count:
        return count - 1
    return <Py_ssize_t>at


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class _Outline:
    """The rings of a valid polygon, each with the area on its left (``following[k]``
    and ``preceding[k]`` the vertices next to vertex ``k`` in its ring), and a grid of
    cells over them, each cell with the edges that come within ``_GRID_MARGIN`` of it.
    Edge ``k`` runs from vertex ``k`` to vertex ``following[k]``."""

    cdef double[:, ::1] vertices
    cdef Py_ssize_t[::1] following
    cdef Py_ssize_t[::1] preceding
    cdef double min_x, min_y, width, height
    cdef Py_ssize_t columns, rows
    cdef Py_ssize_t[::1] cell_start
    cdef Py_ssize_t[::1] cell_edges
    cdef Py_ssize_t[::1] stamp
    cdef Py_ssize_t visit

    def __init__(self, vertices, following, preceding):
        self.vertices = vertices
        self.following = following
        self.preceding = preceding
        cdef Py_ssize_t n = vertices.shape[0]
        self.columns = self.rows = max(1, <Py_ssize_t>sqrt(n / 2.0))
        xy = np.asarray(vertices)
        self.min_x = xy[:, 0].min()
        self.min_y = xy[:, 1].min()
        self.width = (xy[:, 0].max() - self.min_x) / self.columns or 1.0
        self.height = (xy[:, 1].max() - self.min_y) / self.rows or 1.0
        counts_arr = np.zeros(self.columns * self.rows + 1, dtype=np.intp)
        cdef Py_ssize_t[::1] counts = counts_arr
        cdef Py_ssize_t k, c, r
        cdef Py_ssize_t[4] box
        for k in range(n):
            self._edge_cells(k, box)
            for c in range(box[0], box[1] + 1):
                for r in range(box[2], box[3] + 1):
                    counts[c * self.rows + r + 1] += 1
        self.cell_start = np.cumsum(counts_arr)
        self.cell_edges = np.empty(self.cell_start[self.columns * self.rows], dtype=np.intp)
        filled_arr = np.asarray(self.cell_start)[: self.columns * self.rows].copy()
        cdef Py_ssize_t[::1] filled = filled_arr
        for k in range(n):
            self._edge_cells(k, box)
            for c in range(box[0], box[1] + 1):
                for r in range(box[2], box[3] + 1):
                    self.cell_edges[filled[c * self.rows + r]] = k
                    filled[c * self.rows + r] += 1
        self.stamp = np.zeros(n, dtype=np.intp)
        self.visit = 0

    cdef inline Py_ssize_t _cell_column(self, double x) noexcept nogil:
        """The column of cells that holds the longitude ``x``, the nearest one outside."""
        return _cell_index(x, self.min_x, self.width, self.columns)

    cdef inline Py_ssize_t _cell_row(self, double y) noexcept nogil:
        """The row of cells that holds the latitude ``y``, the nearest one outside."""
        return _cell_index(y, self.min_y, self.height, self.rows)

    cdef void _edge_cells(self, Py_ssize_t k, Py_ssize_t* box) noexcept nogil:
        """The first and last column and row of the cells near edge ``k``."""
        cdef Py_ssize_t b = self.following[k]
        cdef double ax = self.vertices[k, 0], ay = self.vertices[k, 1]
        cdef double bx = self.vertices[b, 0], by = self.vertices[b, 1]
        box[0] = self._cell_column((ax if ax < bx else bx) - _GRID_MARGIN)
        box[1] = self._cell_column((bx if ax < bx else ax) + _GRID_MARGIN)
        box[2] = self._cell_row((ay if ay < by else by) - _GRID_MARGIN)
        box[3] = self._cell_row((by if ay < by else ay) + _GRID_MARGIN)

    cdef inline double _edge_distance(self, Py_ssize_t k, double x, double y) noexcept nogil:
        """The squared distance from the point to edge ``k``."""
        cdef Py_ssize_t b = self.following[k]
        return _squared_distance(
            x, y, self.vertices[k, 0], self.vertices[k, 1],
            self.vertices[b, 0], self.vertices[b, 1],
        )

    cdef bint clear_of_rings(self, double x, double y) noexcept nogil:
        """Whether the point lies more than ``_CLEAR`` degrees from every ring."""
        cdef Py_ssize_t c, r, i, k
        cdef double limit = _CLEAR * _CLEAR
        self.visit += 1
        for c in range(self._cell_column(x - _CLEAR), self._cell_column(x + _CLEAR) + 1):
            for r in range(self._cell_row(y - _CLEAR), self._cell_row(y + _CLEAR) + 1):
                for i in range(self.cell_start[c * self.rows + r],
                               self.cell_start[c * self.rows + r + 1]):
                    k = self.cell_edges[i]
                    if self.stamp[k] == self.visit:
                        continue
                    self.stamp[k] = self.visit
                    if self._edge_distance(k, x, y) <= limit:
                        return False
        return True

    cdef bint inside(self, double x, double y) noexcept nogil:
        """Whether a point clear of the rings lies inside the polygon, by the even-odd
        rule along a ray towards growing longitudes."""
        cdef Py_ssize_t c, i, k, b, r = self._cell_row(y)
        cdef double ax, ay, bx, by
        cdef bint inside = False
        self.visit += 1
        for c in range(self._cell_column(x), self.columns):
            for i in range(self.cell_start[c * self.rows + r],
                           self.cell_start[c * self.rows + r + 1]):
                k = self.cell_edges[i]
                if self.stamp[k] == self.visit:
                    continue
                self.stamp[k] = self.visit
                b = self.following[k]
                ax = self.vertices[k, 0]
                ay = self.vertices[k, 1]
                bx = self.vertices[b, 0]
                by = self.vertices[b, 1]
                if (ay > y) != (by > y):
                    if x < ax + (y - ay) * (bx - ax) / (by - ay):
                        inside = not inside
        return inside

    cdef bint clearly_outside(self, double x, double y) noexcept nogil:
        """Whether the point lies outside the polygon and more than ``_CLEAR`` degrees
        from its rings."""
        return self.clear_of_rings(x, y) and not self.inside(x, y)

    cdef int leaves_at(self, Py_ssize_t vertex, double tx, double ty) noexcept nogil:
        """1 when a line from ``vertex`` towards ``t`` starts into the area, -1 when it
        starts out of it, 0 when it runs along a ring or rounding leaves it open."""
        cdef double px = self.vertices[vertex, 0], py = self.vertices[vertex, 1]
        cdef Py_ssize_t a = self.preceding[vertex], b = self.following[vertex]
        cdef double ax = self.vertices[a, 0], ay = self.vertices[a, 1]
        cdef double bx = self.vertices[b, 0], by = self.vertices[b, 1]
        cdef int turn = _orientation(ax, ay, px, py, bx, by)
        cdef int left_in = _orientation(ax, ay, px, py, tx, ty)
        cdef int left_out = _orientation(px, py, bx, by, tx, ty)
        if turn == 0 or left_in == 0 or left_out == 0:
            return 0
        if turn > 0:
            return 1 if left_in > 0 and left_out > 0 else -1
        return 1 if left_in > 0 or left_out > 0 else -1

    cdef bint outside_along(
        self, double px, double py, double dx, double dy, double t, double step
    ) noexcept nogil:
        """Whether a point of the line ``p + s * d`` (``0 < s < 1``) at ``s = t + step``
        times 1, 10 or 100 (``step`` in units of ``s``) lies clearly outside."""
        cdef double s
        cdef int i
        for i in range(3):
            s = t + step
            if not 0 < s < 1:
                return False
            if self.clearly_outside(px + s * dx, py + s * dy):
                return True
            step *= 10
        return False

    cdef int line_status(
        self, double px, double py, Py_ssize_t vp, double qx, double qy, Py_ssize_t vq
    ) noexcept nogil:
        """Whether the straight line from ``p`` to ``q`` (at the vertices ``vp`` and
        ``vq``, -1 for none) crosses or touches a ring away from its ends: ``_VISIBLE``
        when it does not, ``_HIDDEN`` when it crosses into a place clearly outside, else
        ``_OPEN``. A crossing is followed beyond the crossed edge, on its outer side."""
        cdef double dx = qx - px, dy = qy - py
        cdef double low_x = px if px < qx else qx, high_x = qx if px < qx else px
        cdef double low_y = py if py < qy else qy, high_y = qy if py < qy else py
        cdef double slab_low, slab_high, y_low, y_high, ax, ay, bx, by, cross, t
        cdef Py_ssize_t c, r, i, k, b
        cdef int s1, s2, s3, s4, status = _VISIBLE
        self.visit += 1
        cdef Py_ssize_t visit = self.visit
        for c in range(self._cell_column(low_x - _GRID_MARGIN),
                       self._cell_column(high_x + _GRID_MARGIN) + 1):
            # The latitudes of the line within this column of cells
            slab_low = self.min_x + c * self.width
            slab_high = slab_low + self.width
            slab_low = (slab_low if slab_low > low_x else low_x) - _GRID_MARGIN
            slab_high = (slab_high if slab_high < high_x else high_x) + _GRID_MARGIN
            if dx != 0:
                y_low = py + (slab_low - px) * dy / dx
                y_high = py + (slab_high - px) * dy / dx
                if y_low > y_high:
                    y_low, y_high = y_high, y_low
                y_low = y_low if y_low > low_y else low_y
                y_high = y_high if y_high < high_y else high_y
            else:
                y_low = low_y
                y_high = high_y
            for r in range(self._cell_row(y_low - _GRID_MARGIN),
                           self._cell_row(y_high + _GRID_MARGIN) + 1):
                for i in range(self.cell_start[c * self.rows + r],
                               self.cell_start[c * self.rows + r + 1]):
                    k = self.cell_edges[i]
                    if self.stamp[k] == visit:
                        continue
                    self.stamp[k] = visit
                    b = self.following[k]
                    if k == vp or k == vq or b == vp or b == vq:
                        continue
                    ax = self.vertices[k, 0]
                    ay = self.vertices[k, 1]
                    bx = self.vertices[b, 0]
                    by = self.vertices[b, 1]
                    if (ax < low_x and bx < low_x) or (ax > high_x and bx > high_x):
                        continue
                    if (ay < low_y and by < low_y) or (ay > high_y and by > high_y):
                        continue
                    s1 = _orientation(ax, ay, bx, by, px, py)
                    s2 = _orientation(ax, ay, bx, by, qx, qy)
                    if s1 * s2 > 0:
                        continue
                    s3 = _orientation(px, py, qx, qy, ax, ay)
                    s4 = _orientation(px, py, qx, qy, bx, by)
                    if s3 * s4 > 0:
                        continue
                    status = _OPEN
                    if s1 * s2 < 0 and s3 * s4 < 0:
                        cross = dx * (by - ay) - dy * (bx - ax)
                        t = ((ax - px) * (by - ay) - (ay - py) * (bx - ax)) / cross
                        # 3 * _CLEAR from the edge, towards the end on its outer side
                        if self.outside_along(
                            px, py, dx, dy, t,
                            (1 if s2 < 0 else -1) * 3 * _CLEAR
                            * sqrt((bx - ax) ** 2 + (by - ay) ** 2) / fabs(cross),
                        ):
                            return _HIDDEN
        return status


@cython.boundscheck(False)
@cython.wraparound(False)
def visibility(
    double[:, ::1] vertices,
    Py_ssize_t[::1] following,
    Py_ssize_t[::1] preceding,
    const double[:, ::1] points,
    const Py_ssize_t[::1] point_vertex,
):
    """For each pair of ``points`` (``i < j``, in the order of ``np.triu_indices``),
    whether the straight line between them lies inside a valid polygon: ``VISIBLE``,
    ``HIDDEN``, or ``OPEN`` where rounding leaves it open. The polygon's rings are the
    ``vertices``, each ring with the area on its left, ``following[k]`` and
    ``preceding[k]`` the vertices next to ``k`` in its ring; ``point_vertex[i]`` is the
    vertex at point ``i`` (-1 for a point at no vertex)."""
    cdef _Outline outline = _Outline(vertices, following, preceding)
    cdef Py_ssize_t n = points.shape[0]
    cdef Py_ssize_t i, j, end, pair = 0, vi, vj
    cdef double px, py, qx, qy
    cdef int status, leaving
    status_arr = np.empty(n * (n - 1) // 2, dtype=np.uint8)
    clear_arr = np.empty(n, dtype=np.uint8)
    cdef unsigned char[::1] statuses = status_arr
    cdef unsigned char[::1] clear = clear_arr
    with nogil:
        # A point at no vertex counts only when it lies inside, clear of the rings
        for i in range(n):
            clear[i] = (
                point_vertex[i] < 0
                and outline.clear_of_rings(points[i, 0], points[i, 1])
                and outline.inside(points[i, 0], points[i, 1])
            )
        for i in range(n):
            px = points[i, 0]
            py = points[i, 1]
            vi = point_vertex[i]
            for j in range(i + 1, n):
                qx = points[j, 0]
                qy = points[j, 1]
                vj = point_vertex[j]
                status = _VISIBLE
                if vi >= 0 and vj >= 0 and (following[vi] == vj or following[vj] == vi):
                    # The line is an edge of the outline
                    statuses[pair] = status
                    pair += 1
                    continue
                # Where the line starts and ends: into the area, out of it, or open
                for end in range(2):
                    if (vi if end == 0 else vj) < 0:
                        if not clear[i if end == 0 else j]:
                            status = _OPEN
                        continue
                    if end == 0:
                        leaving = outline.leaves_at(vi, qx, qy)
                    else:
                        leaving = outline.leaves_at(vj, px, py)
                    if leaving != 1:
                        status = _OPEN
                    if leaving == -1 and outline.outside_along(
                        px, py, qx - px, qy - py, end,
                        (1 if end == 0 else -1) * _STEP
                        / sqrt((qx - px) ** 2 + (qy - py) ** 2),
                    ):
                        status = _HIDDEN
                        break
                if status != _HIDDEN:
                    leaving = outline.line_status(px, py, vi, qx, qy, vj)
                    if leaving != _VISIBLE:
                        status = leaving
                statuses[pair] = status
                pair += 1
    return status_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def shortest_paths(
    Py_ssize_t n,
    const Py_ssize_t[::1] first,
    const Py_ssize_t[::1] second,
    const double[::1] length,
    double tie,
):
    """The shortest paths between ``n`` points over the straight lines ``first``,
    ``second`` of ``length``, as ``(step, distance)``: ``step[i, j]`` is the point after
    ``i`` on the way to ``j``, -1 where no path joins them, and ``distance[i, j]`` the
    path's length (infinite there). A path replaces another only when it is shorter than
    ``tie`` times its length."""
    distance_arr = np.full((n, n), np.inf)
    step_arr = np.tile(np.arange(n, dtype=np.intp), (n, 1))
    cdef double[:, ::1] distance = distance_arr
    cdef Py_ssize_t[:, ::1] step = step_arr
    cdef Py_ssize_t i, j, k, m = first.shape[0]
    cdef double through, via
    with nogil:
        for i in range(n):
            distance[i, i] = 0.0
        for k in range(m):
            distance[first[k], second[k]] = length[k]
            distance[second[k], first[k]] = length[k]
        for k in range(n):
            for i in range(n):
                via = distance[i, k]
                if via == INFINITY:
                    continue
                for j in range(n):
                    through = via + distance[k, j]
                    if through < distance[i, j] * tie:
                        distance[i, j] = through
                        step[i, j] = step[i, k]
        for i in range(n):
            for j in range(n):
                if distance[i, j] == INFINITY:
                    step[i, j] = -1
    return step_arr, distance_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def entrance_paths(const Py_ssize_t[:, ::1] step, const double[:, ::1] distance,
                   Py_ssize_t count):
    """The paths between the first ``count`` points that ``step`` and ``distance`` (see
    :func:`shortest_paths`) join, pair ``(a, b)`` by pair with ``a < b``, as ``(first,
    second, lengths, offsets, path)``: the points of the path of pair ``i`` are
    ``path[offsets[i]:offsets[i + 1]]``."""
    first, second, lengths, offsets, path = [], [], [], [0], []
    cdef Py_ssize_t a, b, at
    for a in range(count):
        for b in range(a + 1, count):
            if step[a, b] < 0:
                continue
            at = a
            path.append(at)
            while at != b:
                at = step[at, b]
                path.append(at)
            first.append(a)
            second.append(b)
            lengths.append(distance[a, b])
            offsets.append(len(path))
    return (
        np.array(first, dtype=np.intp),
        np.array(second, dtype=np.intp),
        np.array(lengths, dtype=float),
        np.array(offsets, dtype=np.intp),
        np.array(path, dtype=np.intp),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def entrances_passed(
    const double[:, ::1] points,
    Py_ssize_t count,
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] path,
    double tolerance,
):
    """For each path (the points ``path[offsets[i]:offsets[i + 1]]``), how many of the
    first ``count`` points lie within ``tolerance`` degrees of it, its two ends included,
    and whether rounding leaves that open for one of them, as ``(passed, open)``."""
    cdef Py_ssize_t paths = offsets.shape[0] - 1
    cdef Py_ssize_t i, e, s, a, b
    cdef double x, y, d, segment, low_x, high_x, low_y, high_y
    cdef double near = (0.5 * tolerance) ** 2, far = (2.0 * tolerance) ** 2
    cdef double margin = 2.0 * tolerance
    passed_arr = np.zeros(paths, dtype=np.intp)
    open_arr = np.zeros(paths, dtype=np.uint8)
    cdef Py_ssize_t[::1] passed = passed_arr
    cdef unsigned char[::1] undecided = open_arr
    with nogil:
        for i in range(paths):
            a = path[offsets[i]]
            b = path[offsets[i + 1] - 1]
            low_x = high_x = points[a, 0]
            low_y = high_y = points[a, 1]
            for s in range(offsets[i], offsets[i + 1]):
                x = points[path[s], 0]
                y = points[path[s], 1]
                low_x = x if x < low_x else low_x
                high_x = x if x > high_x else high_x
                low_y = y if y < low_y else low_y
                high_y = y if y > high_y else high_y
            for e in range(count):
                if e == a or e == b:
                    passed[i] += 1
                    continue
                x = points[e, 0]
                y = points[e, 1]
                if (x < low_x - margin or x > high_x + margin
                        or y < low_y - margin or y > high_y + margin):
                    continue
                d = far * 2.0
                for s in range(offsets[i], offsets[i + 1] - 1):
                    segment = _squared_distance(
                        x, y, points[path[s], 0], points[path[s], 1],
                        points[path[s + 1], 0], points[path[s + 1], 1],
                    )
                    if segment < d:
                        d = segment
                if d < near:
                    passed[i] += 1
                elif d <= far:
                    undecided[i] = 1
    return passed_arr, open_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _joined_within(
    Py_ssize_t start, Py_ssize_t end, double bound,
    const Py_ssize_t[::1] head, const Py_ssize_t[::1] following,
    const Py_ssize_t[::1] target, const double[::1] weight,
    double[::1] best, Py_ssize_t[::1] seen, Py_ssize_t query,
    double[::1] heap_distance, Py_ssize_t[::1] heap_node,
) noexcept nogil:
    """Whether the kept lines (``head``, ``following``, ``target`` and ``weight``: for
    each point a linked list of its lines) join ``start`` to ``end`` along a route at
    most ``bound`` long. ``best`` holds the shortest known distances of the points whose
    ``seen`` is ``query``; the heap arrays have room for every line twice and a start."""
    cdef Py_ssize_t size = 1, at, line, to, i, child
    cdef double distance, through, d
    cdef Py_ssize_t node
    best[start] = 0.0
    seen[start] = query
    heap_distance[0] = 0.0
    heap_node[0] = start
    while size:
        distance = heap_distance[0]
        at = heap_node[0]
        size -= 1
        # Sift the last entry down from the top
        d = heap_distance[size]
        node = heap_node[size]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= size:
                break
            if child + 1 < size and heap_distance[child + 1] < heap_distance[child]:
                child += 1
            if heap_distance[child] >= d:
                break
            heap_distance[i] = heap_distance[child]
            heap_node[i] = heap_node[child]
            i = child
        heap_distance[i] = d
        heap_node[i] = node
        if at == end:
            return True
        if distance > best[at]:
            continue
        line = head[at]
        while line >= 0:
            to = target[line]
            through = distance + weight[line]
            if through <= bound and (seen[to] != query or through < best[to]):
                best[to] = through
                seen[to] = query
                # Sift the new entry up
                i = size
                size += 1
                while i > 0 and heap_distance[(i - 1) // 2] > through:
                    heap_distance[i] = heap_distance[(i - 1) // 2]
                    heap_node[i] = heap_node[(i - 1) // 2]
                    i = (i - 1) // 2
                heap_distance[i] = through
                heap_node[i] = to
            line = following[line]
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
def spanning_pairs(
    Py_ssize_t n,
    const Py_ssize_t[::1] first,
    const Py_ssize_t[::1] second,
    const double[::1] lengths,
    const Py_ssize_t[::1] order,
    double stretch,
):
    """The indices of the pairs of points ``first``, ``second`` (of ``n`` points, the
    shortest line between them ``lengths`` long) that get an edge, of those in
    ``order``, sorted: taken in that order, a pair is left out when the pairs kept before
    it already join its two points within ``stretch`` times its length."""
    cdef Py_ssize_t m = order.shape[0]
    cdef Py_ssize_t i, k, a, b, lines = 0, kept_count = 0
    head_arr = np.full(n, -1, dtype=np.intp)
    following_arr = np.empty(2 * m, dtype=np.intp)
    target_arr = np.empty(2 * m, dtype=np.intp)
    weight_arr = np.empty(2 * m, dtype=float)
    best_arr = np.empty(n, dtype=float)
    seen_arr = np.full(n, -1, dtype=np.intp)
    heap_distance_arr = np.empty(2 * m + 1, dtype=float)
    heap_node_arr = np.empty(2 * m + 1, dtype=np.intp)
    kept_arr = np.empty(m, dtype=np.intp)
    cdef Py_ssize_t[::1] head = head_arr
    cdef Py_ssize_t[::1] following = following_arr
    cdef Py_ssize_t[::1] target = target_arr
    cdef double[::1] weight = weight_arr
    cdef double[::1] best = best_arr
    cdef Py_ssize_t[::1] seen = seen_arr
    cdef double[::1] heap_distance = heap_distance_arr
    cdef Py_ssize_t[::1] heap_node = heap_node_arr
    cdef Py_ssize_t[::1] kept = kept_arr
    with nogil:
        for k in range(m):
            i = order[k]
            a = first[i]
            b = second[i]
            if _joined_within(
                a, b, stretch * lengths[i], head, following, target, weight,
                best, seen, k, heap_distance, heap_node,
            ):
                continue
            target[lines] = b
            weight[lines] = lengths[i]
            following[lines] = head[a]
            head[a] = lines
            lines += 1
            target[lines] = a
            weight[lines] = lengths[i]
            following[lines] = head[b]
            head[b] = lines
            lines += 1
            kept[kept_count] = i
            kept_count += 1
    return np.sort(kept_arr[:kept_count])
