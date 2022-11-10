import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                os.pardir)
)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.cloner_slam import ClonerSLAM

if __name__ == "__main__":
    cloner_slam = ClonerSLAM("../cfg/default_settings.yaml")

    cloner_slam.Start()

    for x in range(100000):
        cloner_slam.ProcessLidar(x)
        cloner_slam.ProcessRGB(10-x)

    cloner_slam.Join()