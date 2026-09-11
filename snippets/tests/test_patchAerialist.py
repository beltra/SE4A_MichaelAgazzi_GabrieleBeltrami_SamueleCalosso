"""The startup patches must be safe to apply twice.

applyPatches runs from cli.py on every campaign and again at image build time,
so each patch is applied to files that may already carry it. A patch that is
not idempotent doubles its own edit, and the damage only shows up much later
as a simulator that will not arm. Every patch is exercised against a copy of
the real anchor text, never against the installed Aerialist.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import patchAerialist as pa


LAUNCH_FILE = (
    '<launch>\n'
    '    <arg name="vehicle" default="iris"/>\n'
    + pa.LAUNCH_OLD +
    '\n</launch>\n'
)

SIMULATOR_FILE = (
    "class Simulator:\n"
    "    def run(self):\n"
    '        sim_command = ""\n'
    + pa.SIM_ANCHOR + ' {self.world}"\n'
)

DRONE_FILE = (
    "    async def connect_async(self):\n"
    + pa.DRONE_OLD +
    "                break\n"
)


class PatchCase(unittest.TestCase):
    """Runs one patch function against a temporary copy of its target file."""

    constant = ""
    function = ""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "target"

    def applyTo(self, text):
        """Write text to the target, run the patch, return (status, content)."""
        self.path.write_text(text)
        with patch.object(pa, self.constant, str(self.path)):
            status = self.patchFunction()
        return status, self.path.read_text()

    def patchFunction(self):
        return getattr(pa, self.function)()


class LaunchPatchTests(PatchCase):
    constant, function = "LAUNCH_PATH", "patchLaunch"

    def test_a_fresh_file_gains_the_non_interactive_argument(self):
        status, text = self.applyTo(LAUNCH_FILE)
        self.assertEqual(status, "launch: patched")
        self.assertIn('<arg name="interactive" value="false"/>', text)

    def test_applying_it_twice_changes_nothing(self):
        _, once = self.applyTo(LAUNCH_FILE)
        status, twice = self.applyTo(once)
        self.assertEqual(status, "launch: already patched")
        self.assertEqual(once, twice)
        self.assertEqual(twice.count('name="interactive"'), 1)

    def test_an_unrecognised_file_is_left_alone(self):
        status, text = self.applyTo("<launch>\n</launch>\n")
        self.assertEqual(status, "launch: anchor not found")
        self.assertEqual(text, "<launch>\n</launch>\n")

    def test_a_missing_file_is_reported_not_created(self):
        with patch.object(pa, "LAUNCH_PATH", str(self.path / "absent")):
            self.assertTrue(pa.patchLaunch().startswith("launch: missing"))
        self.assertFalse((self.path / "absent").exists())


class SimulatorPatchTests(PatchCase):
    constant, function = "SIMULATOR_PATH", "patchSimulator"

    def test_a_fresh_file_gains_the_speed_factor_export(self):
        status, text = self.applyTo(SIMULATOR_FILE)
        self.assertEqual(status, "simulator: patched")
        self.assertIn("PX4_SIM_SPEED_FACTOR", text)
        # The export has to precede the roslaunch it applies to.
        self.assertLess(text.index("PX4_SIM_SPEED_FACTOR"), text.index("exec roslaunch"))

    def test_applying_it_twice_does_not_export_twice(self):
        _, once = self.applyTo(SIMULATOR_FILE)
        status, twice = self.applyTo(once)
        self.assertEqual(status, "simulator: already patched")
        self.assertEqual(once, twice)
        self.assertEqual(twice.count("PX4_SIM_SPEED_FACTOR"), 1)

    def test_only_the_first_anchor_is_exported_before(self):
        # simulator.py launches the sim from more than one branch; the export
        # belongs to the ROS one alone, so the replace is deliberately capped.
        _, text = self.applyTo(SIMULATOR_FILE + "\n" + pa.SIM_ANCHOR + ' {self.other}"\n')
        self.assertEqual(text.count("PX4_SIM_SPEED_FACTOR"), 1)

    def test_an_unrecognised_file_is_left_alone(self):
        status, text = self.applyTo("class Simulator:\n    pass\n")
        self.assertEqual(status, "simulator: anchor not found")
        self.assertEqual(text, "class Simulator:\n    pass\n")


class DronePatchTests(PatchCase):
    constant, function = "DRONE_PATH", "patchDrone"

    def test_a_fresh_file_also_waits_for_the_home_position(self):
        status, text = self.applyTo(DRONE_FILE)
        self.assertEqual(status, "drone: patched")
        self.assertIn("is_global_position_ok and is_home_position_ok", text.replace("global_lock.", ""))

    def test_applying_it_twice_does_not_repeat_the_condition(self):
        _, once = self.applyTo(DRONE_FILE)
        status, twice = self.applyTo(once)
        self.assertEqual(status, "drone: already patched")
        self.assertEqual(once, twice)
        self.assertEqual(twice.count("is_home_position_ok"), 1)

    def test_an_unrecognised_file_is_left_alone(self):
        status, text = self.applyTo("async def connect_async(self):\n    pass\n")
        self.assertEqual(status, "drone: anchor not found")
        self.assertEqual(text, "async def connect_async(self):\n    pass\n")


class ApplyPatchesTests(unittest.TestCase):
    def test_every_patch_is_attempted_and_reported(self):
        with patch.object(pa, "patchLaunch", return_value="launch: patched") as launch, \
                patch.object(pa, "patchSimulator", return_value="simulator: patched") as simulator, \
                patch.object(pa, "patchDrone", return_value="drone: patched") as drone, \
                patch("builtins.print") as printed:
            pa.applyPatches()

        for step in (launch, simulator, drone):
            step.assert_called_once_with()
        self.assertEqual([call.args[0] for call in printed.call_args_list],
                         ["launch: patched", "simulator: patched", "drone: patched"])


if __name__ == "__main__":
    unittest.main()
