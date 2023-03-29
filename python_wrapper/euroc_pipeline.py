import argh
from datasets.euroc import EUROCDataset
from orb_slam3 import ORB_SLAM3


def main(
    data_source: str,
    config: str,
    vocabulary: str,
    sensor: str = "monocular",
    visualize: bool = True,
):
    dataset = EUROCDataset(data_source)
    slam = ORB_SLAM3(vocabulary, config, sensor, visualize)  # initialize
    for value in dataset:
        imL, imR, timestamp = value
        if sensor == "monocular":
            slam.TrackMonocular(imL, timestamp)
        if sensor == "stereo":
            slam.TrackStereo(imL, imR, timestamp)
    slam.SaveTrajectoryTUM(f"Euroc_tum_{sensor}.txt")
    slam.SaveTrajectoryEuroC(f"EuroC_{sensor}.txt")
    slam.Shutdown()


if __name__ == "__main__":
    argh.dispatch_command(main)
