"""Idempotent patches applied at container startup.

Both patches must run before any Aerialist simulator is built:
  1. PX4 SITL would block on stdin without `interactive:=false`, so subprocess
     launches exit immediately.
  2. Aerialist's ROS branch does not export `PX4_SIM_SPEED_FACTOR`, which makes
     `simulation.speed` from the mission YAML a no-op for that backend.
"""

import os
import sys


LAUNCH_PATH = "/src/aerialist/aerialist/resources/simulation/collision_prevention.launch"
SIMULATOR_PATH = "/src/aerialist/aerialist/px4/simulator.py"

LAUNCH_OLD = (
    '    <include file="$(find px4)/launch/px4.launch">\n'
    '        <arg name="vehicle" value="$(arg vehicle)"/>\n'
    "    </include>"
)
LAUNCH_NEW = (
    '    <include file="$(find px4)/launch/px4.launch">\n'
    '        <arg name="vehicle" value="$(arg vehicle)"/>\n'
    '        <arg name="interactive" value="false"/>\n'
    "    </include>"
)

SIM_ANCHOR = '            sim_command += f"exec roslaunch {self.AVOIDANCE_LAUNCH}'
SIM_MARKER = 'sim_command += f"export PX4_SIM_SPEED_FACTOR={self.config.speed};'
SIM_EXPORT = (
    "            if self.config.speed != 1:\n"
    '                sim_command += f"export PX4_SIM_SPEED_FACTOR={self.config.speed}; "\n'
)


def patchLaunch() -> str:
    if not os.path.exists(LAUNCH_PATH):
        return f"launch: missing ({LAUNCH_PATH})"
    with open(LAUNCH_PATH) as fh:
        text = fh.read()
    if 'interactive" value="false"' in text:
        return "launch: already patched"
    if LAUNCH_OLD not in text:
        return "launch: anchor not found"
    with open(LAUNCH_PATH, "w") as fh:
        fh.write(text.replace(LAUNCH_OLD, LAUNCH_NEW))
    return "launch: patched"


def patchSimulator() -> str:
    if not os.path.exists(SIMULATOR_PATH):
        return f"simulator: missing ({SIMULATOR_PATH})"
    with open(SIMULATOR_PATH) as fh:
        text = fh.read()
    if SIM_MARKER in text:
        return "simulator: already patched"
    if SIM_ANCHOR not in text:
        return "simulator: anchor not found"
    with open(SIMULATOR_PATH, "w") as fh:
        fh.write(text.replace(SIM_ANCHOR, SIM_EXPORT + SIM_ANCHOR, 1))
    return "simulator: patched"


def applyPatches() -> None:
    print(patchLaunch())
    print(patchSimulator())


if __name__ == "__main__":
    applyPatches()
    sys.exit(0)
