from src.common.sensors import Image, LidarScan
from src.cloner_slam import ClonerSLAM
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


if __name__ == "__main__":
    cloner_slam = ClonerSLAM("../cfg/default_settings.yaml")

    cloner_slam.Start()

    for x in range(1000):
        cloner_slam.process_lidar(LidarScan(timestamps=torch.Tensor([x])))
        cloner_slam.ProcessRGB(Image(3, x))
    cloner_slam.Stop()
