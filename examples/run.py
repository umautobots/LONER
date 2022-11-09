import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                os.pardir)
)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")

from src.cloner_slam import ClonerSLAM
