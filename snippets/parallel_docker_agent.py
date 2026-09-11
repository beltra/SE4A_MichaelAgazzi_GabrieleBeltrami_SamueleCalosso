"""Small hardened wrapper around Aerialist's per-test Docker agent."""

import logging
import os
import subprocess

from decouple import config
from aerialist.px4.docker_agent import DockerAgent


logger = logging.getLogger(__name__)


class ParallelDockerAgent(DockerAgent):
    """Run one simulator in an isolated, disposable worker container."""

    DOCKER_IMG = config("DOCKER_IMG", default="docker-uav-testing")
    USE_GPU = config("SIM_WORKER_GPU", default=True, cast=bool)
    # Docker inside an LXC container cannot apply its default AppArmor
    # profile and every worker fails to start; SIM_WORKER_SECURITY_OPT
    # ("apparmor=unconfined" there) is passed on to `docker run`. Empty
    # everywhere else, so the normal profile stays in force.
    SECURITY_OPT = config("SIM_WORKER_SECURITY_OPT", default="")
    # The avoidance depth camera needs an OpenGL context even in headless mode.
    # A private Xvfb display gives every worker its own context and display ID.
    CPU_CMD = (
        "xvfb-run -a -s '-screen 0 1280x720x24' "
        "env LIBGL_ALWAYS_INDIRECT=0 LIBGL_ALWAYS_SOFTWARE=1 "
        "aerialist exec --test {test_file}"
    )
    GPU_CMD = "aerialist exec --test {test_file}"
    CMD = GPU_CMD if USE_GPU else CPU_CMD
    # Aerialist's default uses `docker exec -it`, which fails when the parent
    # process captures output without an interactive terminal.
    DOCKER_CMD = "docker exec {id} {cmd}"

    def create_container(self):
        command = [
            "docker",
            "run",
            "--rm",
            "-td",
            "--shm-size=2g",
        ]
        if self.SECURITY_OPT:
            command.extend(["--security-opt", self.SECURITY_OPT])
        if self.USE_GPU:
            command.extend(
                [
                    "--gpus",
                    "all",
                    "-e",
                    f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
                    "-e",
                    "XAUTHORITY=/tmp/.Xauthority",
                    "-e",
                    "NVIDIA_DRIVER_CAPABILITIES=all,graphics,display,compute,utility",
                    "-e",
                    "__NV_PRIME_RENDER_OFFLOAD=1",
                    "-e",
                    "__GLX_VENDOR_LIBRARY_NAME=nvidia",
                    "-v",
                    "/tmp/.X11-unix:/tmp/.X11-unix:rw",
                ]
            )
        command.append(self.DOCKER_IMG)
        created = subprocess.run(command, capture_output=True, text=True)
        if created.returncode != 0:
            raise RuntimeError(created.stderr.strip() or "failed to create simulator worker")
        self.container_id = created.stdout.strip()
        if self.USE_GPU:
            copied = subprocess.run(
                [
                    "docker",
                    "cp",
                    os.environ.get("XAUTHORITY", "/tmp/.Xauthority"),
                    f"{self.container_id}:/tmp/.Xauthority",
                ],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                subprocess.run(
                    ["docker", "rm", "-f", self.container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                raise RuntimeError(copied.stderr.strip() or "failed to copy worker Xauthority")
        logger.info("new simulator worker: %s", self.container_id[:12])
        return True

    def run(self):
        try:
            executed = subprocess.run(
                self.docker_cmd,
                shell=True,
                capture_output=True,
                text=True,
            )
            stdout = executed.stdout
            if "LOG:" not in stdout and "logging started:" not in stdout:
                found = subprocess.run(
                    [
                        "docker",
                        "exec",
                        self.container_id,
                        "find",
                        "/root/.ros/log",
                        "-type",
                        "f",
                        "-name",
                        "*.ulg",
                        "-printf",
                        "%T@ %p\\n",
                    ],
                    capture_output=True,
                    text=True,
                )
                logs = []
                for line in found.stdout.splitlines():
                    timestamp, separator, path = line.partition(" ")
                    if separator:
                        logs.append((float(timestamp), path))
                if logs:
                    stdout += f"\nlogging started:{max(logs)[1]}\n"
            if executed.returncode != 0:
                logger.error(
                    "worker command failed with status %d: %s",
                    executed.returncode,
                    executed.stderr.strip(),
                )
            self.process_output(
                executed.returncode,
                stdout,
                executed.stderr,
                True,
            )
            logger.info("worker test execution finished")
            return self.results
        finally:
            # Aerialist returns early when it cannot locate a log path, leaving
            # the worker alive. Always remove it, including on parse failures.
            if getattr(self, "container_id", None):
                subprocess.run(
                    ["docker", "rm", "-f", self.container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
