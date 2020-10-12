import os
import socket

from pathlib import Path


ALL_CA = [
    'ball', 'spoon', 'mug', 'cup',
    'giraffe', 'horse', 'cat', 'duck',
    'helicopter', 'airplane', 'truck', 'car'
]
ALL_NO = list(range(1, 31))
TEST_NO = {
    'ball': [1, 7, 9],
    'spoon': [5, 7, 8],
    'mug': [12, 13, 14],
    'cup': [12, 13, 15],
    'giraffe': [1, 5, 13],
    'horse': [1, 10, 15],
    'cat': [4, 9, 15],
    'duck': [5, 9, 13],
    'helicopter': [5, 10, 15],
    'airplane': [2, 6, 15],
    'truck': [2, 6, 8],
    'car': [6, 11, 13],
}


ALL_TR = [
    'absent', 'present', 'hodgepodge',
    'rxminus', 'rxplus', 'ryminus', 'ryplus', 'rzminus', 'rzplus',
    'tx', 'ty', 'tz'
]
ALL_6FPS = list(range(1, 500))
ALL_1FPS = list(range(1, 500, 6))

ToyboxMeanStdCacheFile = 'mean_std_cache.pickle'


ProjDir = Path(__file__).parent

# ToyboxRaw = '/media/tengyu/DataU/Data/Toybox/frame6_1920_1080/*/*.jpeg'
# ToyboxSquaredCropped = '/media/tengyu/DataU/Data/Toybox/cropped_square/*/*.jpeg'
# ToyboxCropped = '/media/tengyu/DataU/Data/Toybox/cropped/*/*.jpeg'
#
# ToyboxSquares = '/media/tengyu/DataU/Data/Toybox/squares'
ToyboxSquaresRz = '/media/tengyu/DataU/Data/Toybox/squares_rz'

HostName = socket.gethostname()
ToyboxDataDirs = {
    # 'VUSE-103978002': '/home/mat/Data/v2/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # Lab Titan X, 10.20.141.250
    # 'ENG-AIVASLAB1': '/home/mat/Data/v2/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # My Lab 1060, 10.20.141.40
    # 'VUSE-10397': '/home/mat/Data/v2/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # Ryan Lab 1060, 10.20.141.186
    'tengyu-ubuntu': '/media/tengyu/DataU/Data/Toybox/squares_rz',  # Home
}

ToyboxLogDirs = {
    # 'VUSE-103978002': '/home/mat/Log/V2Exp',  # Lab Titan X, 10.20.141.250
    # 'ENG-AIVASLAB1': '/home/mat/Log/V2Exp',  # My Lab 1060, 10.20.141.40
    # 'VUSE-10397': '/home/mat/Log/V2Exp',  # Ryan Lab 1060, 10.20.141.186
    'tengyu-ubuntu': '/media/tengyu/DataU/Log/ToyboxExp',  # Home
}

ToyboxDataDir = ToyboxDataDirs[HostName]
ToyboxLogDir = ToyboxLogDirs[HostName]
