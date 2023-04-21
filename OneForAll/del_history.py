import os
import shutil


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

remove_dir('OneForAll/outputs/vehicle')
remove_dir('OneForAll/outputs/pedestrian')
remove_dir('OneForAll/logs/vehicle')
remove_dir('OneForAll/logs/pedestrian')
