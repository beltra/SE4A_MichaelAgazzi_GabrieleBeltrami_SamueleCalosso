"""Stand-ins for the Aerialist classes, so the tests run without ROS or PX4.

Only the attributes the generator reads are reproduced: an Obstacle holds a
Size and a Position, both plain named tuples like the originals.
"""

import sys
import types
from typing import NamedTuple


class Size(NamedTuple):
    l: float
    w: float
    h: float


class Position(NamedTuple):
    x: float
    y: float
    z: float
    r: float


class Obstacle:
    Size = Size
    Position = Position

    def __init__(self, size, position):
        self.size = size
        self.position = position

    def __repr__(self):
        return "Obstacle(%r, %r)" % (self.size, self.position)


def installAerialistStubs():
    """Register the stub modules; safe to call from every test file."""
    if "aerialist.px4.obstacle" in sys.modules:
        return
    try:
        # Inside the simulator image the real library is there; shadowing it
        # would break the lab tooling tests that run in the same session.
        import aerialist.px4.obstacle  # noqa: F401
        return
    except ImportError:
        pass
    aerialist = types.ModuleType("aerialist")
    px4 = types.ModuleType("aerialist.px4")
    aerialistTest = types.ModuleType("aerialist.px4.aerialist_test")
    obstacle = types.ModuleType("aerialist.px4.obstacle")
    testcase = types.ModuleType("testcase")

    aerialistTest.AerialistTest = type("AerialistTest", (), {})
    obstacle.Obstacle = Obstacle
    testcase.TestCase = type("TestCase", (), {})
    testcase.is_isolated_execution = lambda: False

    sys.modules["aerialist"] = aerialist
    sys.modules["aerialist.px4"] = px4
    sys.modules["aerialist.px4.aerialist_test"] = aerialistTest
    sys.modules["aerialist.px4.obstacle"] = obstacle
    sys.modules["testcase"] = testcase
