from pathlib import Path

class MODEL_CONSTANT:
    DELETION_THRESHOLD = 0.02
    MINIMAL_AGE = 0.01
    SUFFICIENT_STABILIZATION= 0.01

    INPUT_ADAPTATION_THRESHOLD = 0.02
    # LEARNING_RATE_FOR_BEST_NODE = 0.15
    # LEARNING_RATE_FOR_NEIGHBOR = 0.01
    LEARNING_RATE_FOR_BEST_NODE = 0.1
    LEARNING_RATE_FOR_NEIGHBOR = 0.003
    OUTPUT_ADAPTATION_THRESHOLD = -0.03
    OUTPUT_ADAPTATION_LEARNING_RATE = 0.15
    INSERTION_TOLERANCE = 0.005
    INSERTION_LEARNING_RATE = 0.2

    TL = 20
    TS = 5
    TY = 100
    TV = 100

    MAXIMUM_EDGE_AGE = 50
    MAXIMUM_NEW_NODE_AGE = 150

    THETA = 5

    CAPTURE_TIME = 100


DATA_ROOT = Path(r"E:\Coronary_Heart_Disease_Detection\Data\Disease_dataset")

DATASET = {
    "DatasetA": str(DATA_ROOT / "OfficialDatasetA" / "NumpyData" / "train"),
    "DatasetB": str(DATA_ROOT / "OfficialDatasetB" / "NumpyData" / "train"),
    "Env1": str(DATA_ROOT / "Env1" / "NumpyData"),
    "Env2": str(DATA_ROOT / "Env2" / "NumpyData"),
    "Env3": str(DATA_ROOT / "Env3" / "NumpyData"),
    "Env4": str(DATA_ROOT / "Env4" / "NumpyData"),
    "Eval": str(DATA_ROOT / "Eval" / "NumpyData"),
    "RandomDataset": str(DATA_ROOT / "RandomDataset" / "NumpyData"),
}

