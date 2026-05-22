#!/usr/bin/python3
from argparse import ArgumentParser
from datetime import datetime
import logging
import os
import shutil
import sys
from decouple import config

GENERATOR = "ga"  # "ga" or "random"

if GENERATOR == "ga":
    from geneticAlgorithm.generator import GeneticGenerator as Generator
elif GENERATOR == "random":
    from random_generator import RandomGenerator as Generator
else:
    raise ValueError(f"unknown generator: {GENERATOR!r}")

TESTS_FOLDER = config("TESTS_FOLDER", default="./generated_tests/")
logger = logging.getLogger(__name__)


def arg_parse():
    main_parser = ArgumentParser(
        description="UAV Test Generator",
    )
    subparsers = main_parser.add_subparsers()
    parser = subparsers.add_parser(name="generate", description="generate tests")
    parser.add_argument("test", help="initial test description file address")

    parser.add_argument(
        "budget",
        type=int,
        help="test generation budget (total number of simulations allowed)",
    )

    args = main_parser.parse_args()
    return args


def config_loggers():
    os.makedirs("logs/", exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logs/debug.txt",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    root = logging.getLogger()
    # terminal logs
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    root.addHandler(c_handler)

    # file logs
    f_handler = logging.FileHandler("logs/info.txt")
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    root.addHandler(f_handler)


def _ensure_virtual_display() -> None:
    """Start Xvfb when DISPLAY is remote or unset.

    On Windows, DISPLAY is forwarded as host.docker.internal:0.0 (a remote X11
    address). Gazebo's depth-camera plugin needs a real OpenGL context; with
    LIBGL_ALWAYS_INDIRECT the plugin either hangs or fails to initialise, which
    prevents PX4 from completing its prearm checks, causing arm() to return
    COMMAND_DENIED. Starting a local Xvfb with software Mesa avoids the issue.
    """
    import subprocess
    display = os.environ.get("DISPLAY", "")
    if display.startswith(":"):
        return
    if os.environ.get("AERIALIST_VIRTUAL_DISPLAY") == "started":
        return
    try:
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x720x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = ":99"
        os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["AERIALIST_VIRTUAL_DISPLAY"] = "started"
        print("virtual display: started Xvfb on :99 with software OpenGL")
    except FileNotFoundError:
        print("virtual display: Xvfb not found, keeping DISPLAY=" + display)


if __name__ == "__main__":
    import patchAerialist
    patchAerialist.applyPatches()
    _ensure_virtual_display()
    config_loggers()
    try:
        args = arg_parse()
        generator = Generator(args.test)
        test_cases = generator.generate(args.budget)

        ### copying the test cases to the output folder
        tests_fld = f'{TESTS_FOLDER}{datetime.now().strftime("%d-%m-%H-%M-%S")}/'
        os.makedirs(tests_fld, exist_ok=True)
        for i in range(len(test_cases)):
            test_cases[i].save_yaml(f"{tests_fld}/test_{i}.yaml")
            shutil.copy2(test_cases[i].log_file, f"{tests_fld}/test_{i}.ulg")
            shutil.copy2(test_cases[i].plot_file, f"{tests_fld}/test_{i}.png")
        print(f"{len(test_cases)} test cases generated")
        print(f"output folder: {tests_fld}")

    except Exception as e:
        logger.exception("program terminated:" + str(e), exc_info=True)
        sys.exit(1)
