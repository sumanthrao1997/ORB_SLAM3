import argh
from datasets.tum import TUMDataset
from orb_slam3 import ORB_SLAM3


def main(data_source: str, config: str, vocabulary: str, visualize: bool = True):
    dataset = TUMDataset(data_source)
    slam = ORB_SLAM3(vocabulary, config, "rgbd", visualize)  # initialize
    for value in dataset:
        rgb, depth, current_timestamp = value
        slam.TrackRGBD(image=rgb, depthmap=depth, timestamp=current_timestamp)

    slam.SaveTrajectoryTUM("tum_trajectory.txt")
    slam.Shutdown()


if __name__ == "__main__":
    argh.dispatch_command(main)
