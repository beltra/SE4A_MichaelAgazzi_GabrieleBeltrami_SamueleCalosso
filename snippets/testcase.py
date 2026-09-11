import copy
import logging
import threading
import uuid
from typing import List
from decouple import config
from aerialist.px4.aerialist_test import AerialistTest, AgentConfig
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.plot import Plot
from aerialist.px4.trajectory import Trajectory

AGENT = config("AGENT", default=AgentConfig.LOCAL)
if AGENT == AgentConfig.LOCAL:
    from aerialist.px4.local_agent import LocalAgent
if AGENT == AgentConfig.DOCKER:
    from parallel_docker_agent import ParallelDockerAgent
if AGENT == AgentConfig.K8S:
    from aerialist.px4.k8s_agent import K8sAgent

logger = logging.getLogger(__name__)
PLOT_LOCK = threading.Lock()


def is_isolated_execution():
    return AGENT in (AgentConfig.DOCKER, AgentConfig.K8S)


class TestCase(object):
    def __init__(self, casestudy: AerialistTest, obstacles: List[Obstacle]):
        self.test = copy.deepcopy(casestudy)
        self.test.simulation.obstacles = obstacles
        if AGENT == AgentConfig.DOCKER:
            worker_speed = config("SIM_WORKER_SPEED", default="")
            if worker_speed:
                self.test.simulation.speed = min(
                    self.test.simulation.speed, float(worker_speed)
                )
                if self.test.mission is not None:
                    self.test.mission.speed = self.test.simulation.speed
            self.test.agent = AgentConfig(
                engine=AgentConfig.DOCKER,
                path=config("LOGS_COPY_DIR", default="results/logs/"),
                id=f"ga-{uuid.uuid4().hex[:12]}",
            )

    def execute(self) -> Trajectory:
        if AGENT == AgentConfig.LOCAL:
            agent = LocalAgent(self.test)
        if AGENT == AgentConfig.DOCKER:
            agent = ParallelDockerAgent(self.test)
        if AGENT == AgentConfig.K8S:
            agent = K8sAgent(self.test)
        logger.info("running the test...")
        self.test_results = agent.run()
        logger.info("test finished...")
        self.trajectory = self.test_results[0].record
        self.log_file = self.test_results[0].log_file
        return self.trajectory

    def get_distances(self) -> List[float]:
        return [
            self.trajectory.min_distance_to_obstacles([obst])
            for obst in self.test.simulation.obstacles
        ]

    def plot(self):
        # Matplotlib has process-global state and is not thread-safe. Simulator
        # execution stays parallel; only the short plot operation is serialized.
        # Aerialist names plots by the second they were saved, so two tests
        # finishing together would silently share one file. Name it ourselves.
        with PLOT_LOCK:
            self.plot_file = Plot.plot_test(
                self.test, self.test_results, filename=f"plot-{uuid.uuid4().hex[:12]}"
            )

    def save_yaml(self, path):
        self.test.to_yaml(path)
