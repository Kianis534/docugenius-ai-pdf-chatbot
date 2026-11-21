"""Sketch clean-up: turn rough freehand strokes into clean geometric shapes.

Each stroke is classified with a few cheap geometric features:

* **closure** - distance between the first and last point, relative to size
* **straightness** - chord length divided by path length
* **radial variance** - how evenly the points sit around the centroid

Closed strokes become circles or rectangles, open and nearly straight strokes
become lines, and everything else is smoothed with Chaikin's algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean, pstdev
from typing import Literal

ShapeType = Literal["line", "circle", "rectangle", "curve"]

CIRCLE_SEGMENTS = 48
STRAIGHTNESS_THRESHOLD = 0.92
CIRCLE_RADIAL_CV = 0.2


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class EnhancedStroke:
    type: ShapeType
    points: list[Point]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _path_length(points: list[Point]) -> float:
    return sum(_distance(a, b) for a, b in pairwise(points))


def _chaikin(points: list[Point], iterations: int = 2) -> list[Point]:
    for _ in range(iterations):
        if len(points) < 3:
            return points
        smoothed = [points[0]]
        for a, b in pairwise(points):
            smoothed.append(Point(0.75 * a.x + 0.25 * b.x, 0.75 * a.y + 0.25 * b.y))
            smoothed.append(Point(0.25 * a.x + 0.75 * b.x, 0.25 * a.y + 0.75 * b.y))
        smoothed.append(points[-1])
        points = smoothed
    return points


def classify_stroke(points: list[Point]) -> ShapeType:
    if len(points) < 3:
        return "line"
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    diagonal = math.hypot(width, height)
    if diagonal < 1:
        return "curve"

    path = _path_length(points)
    closed = _distance(points[0], points[-1]) <= max(20.0, 0.2 * diagonal) and path > 2 * diagonal
    if not closed:
        straightness = _distance(points[0], points[-1]) / path if path else 1.0
        return "line" if straightness >= STRAIGHTNESS_THRESHOLD else "curve"

    cx, cy = fmean(xs), fmean(ys)
    radii = [math.hypot(p.x - cx, p.y - cy) for p in points]
    mean_radius = fmean(radii)
    radial_cv = pstdev(radii) / mean_radius if mean_radius else 1.0
    aspect = width / height if height else math.inf
    if radial_cv <= CIRCLE_RADIAL_CV and 0.75 <= aspect <= 1.33:
        return "circle"
    return "rectangle"


def enhance_stroke(points: list[Point]) -> EnhancedStroke:
    if not points:
        return EnhancedStroke("curve", [])
    shape = classify_stroke(points)

    if shape == "line":
        return EnhancedStroke("line", [points[0], points[-1]])

    if shape == "circle":
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        radius = fmean(math.hypot(p.x - cx, p.y - cy) for p in points)
        circle = [
            Point(
                cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
                cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS),
            )
            for i in range(CIRCLE_SEGMENTS + 1)
        ]
        return EnhancedStroke("circle", circle)

    if shape == "rectangle":
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        return EnhancedStroke(
            "rectangle",
            [Point(left, top), Point(right, top), Point(right, bottom), Point(left, bottom), Point(left, top)],
        )

    return EnhancedStroke("curve", _chaikin(points))


def enhance_strokes(strokes: Iterable[list[Point]]) -> list[EnhancedStroke]:
    return [enhance_stroke(stroke) for stroke in strokes]
