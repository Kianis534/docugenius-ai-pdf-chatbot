"""Unit tests for sketch shape recognition."""

from __future__ import annotations

import math

from docugenius.services.sketch import Point, classify_stroke, enhance_stroke


def circle(cx=100.0, cy=100.0, r=60.0, n=60, wobble=0.0):
    return [
        Point(
            cx + (r + wobble * math.sin(i)) * math.cos(2 * math.pi * i / n),
            cy + (r + wobble * math.sin(i)) * math.sin(2 * math.pi * i / n),
        )
        for i in range(n + 1)
    ]


def rectangle(x=10.0, y=10.0, w=200.0, h=80.0, step=5.0):
    pts = []
    for i in range(int(w / step)):
        pts.append(Point(x + i * step, y))
    for i in range(int(h / step)):
        pts.append(Point(x + w, y + i * step))
    for i in range(int(w / step)):
        pts.append(Point(x + w - i * step, y + h))
    for i in range(int(h / step) + 1):
        pts.append(Point(x, y + h - i * step))
    return pts


def test_detects_circle_even_when_wobbly():
    assert classify_stroke(circle()) == "circle"
    assert classify_stroke(circle(wobble=4)) == "circle"


def test_detects_rectangle():
    assert classify_stroke(rectangle()) == "rectangle"


def test_detects_line_and_curve():
    line = [Point(x, 50 + (x % 3)) for x in range(0, 300, 10)]
    arc = [Point(100 + 80 * math.cos(t / 20), 100 + 80 * math.sin(t / 20)) for t in range(0, 40)]
    assert classify_stroke(line) == "line"
    assert classify_stroke(arc) == "curve"


def test_enhanced_circle_is_closed_and_round():
    result = enhance_stroke(circle(wobble=3))
    assert result.type == "circle"
    assert result.points[0] == result.points[-1] or math.isclose(result.points[0].x, result.points[-1].x)
    cx = sum(p.x for p in result.points[:-1]) / (len(result.points) - 1)
    cy = sum(p.y for p in result.points[:-1]) / (len(result.points) - 1)
    radii = [math.hypot(p.x - cx, p.y - cy) for p in result.points]
    assert max(radii) - min(radii) < 0.01
    assert 55 < radii[0] < 65


def test_enhanced_rectangle_uses_bounding_box():
    result = enhance_stroke(rectangle(x=10, y=20, w=100, h=50))
    assert result.type == "rectangle"
    assert result.points[0] == result.points[-1] == Point(10, 20)
    assert Point(110, 70) in result.points or any(abs(p.x - 105) < 6 and abs(p.y - 70) < 6 for p in result.points)


def test_degenerate_inputs():
    assert enhance_stroke([]).points == []
    assert enhance_stroke([Point(1, 1), Point(5, 5)]).type == "line"
    assert enhance_stroke([Point(1, 1), Point(1, 1), Point(1, 1)]).type == "curve"
