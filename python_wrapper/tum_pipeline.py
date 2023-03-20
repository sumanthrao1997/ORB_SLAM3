import time
import argh
from tum import TUMDataset
from python_wrapper import orb_slam_pybind as orb
import numpy as np
from pyquaternion.quaternion import Quaternion
import contextlib


def main(data_source: str, config: str, vocabulary: str):
    dataset = TUMDataset(data_source)
    timestamps = dataset.get_timestamps()
    gt_poses = dataset.gt_poses
    execution_times = []
    slam = orb.System(vocabulary, config, orb.System.eSensor.RGBD, True)
    for idx, value in enumerate(dataset):
        rgb, depth, current_timestamp = value
        start_time = time.time()
        slam.TrackRGBD(rgb, depth, current_timestamp)
        end_time = time.time()
        execution_times.append(end_time - start_time)

    slam.SaveTrajectoryTUM("bullshit.txt")
    save_poses_kitti_format("gt_tum1", gt_poses)
    save_poses_tum_format("gt_tum1", gt_poses, timestamps)
    slam.Shutdown()


def save_poses_kitti_format(filename: str, poses):
    def _to_kitti_format(poses: np.ndarray) -> np.ndarray:
        return np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])

    np.savetxt(fname=f"{filename}_kitti.txt", X=_to_kitti_format(poses))


def save_poses_tum_format(filename, poses, timestamps):
    def _to_tum_format(poses, timestamps):
        tum_data = []
        with contextlib.suppress(ValueError):
            for idx in range(len(poses)):
                tx, ty, tz = poses[idx][:3, -1].flatten()
                qw, qx, qy, qz = Quaternion(matrix=poses[idx], atol=0.01).elements
                tum_data.append([timestamps[idx], tx, ty, tz, qx, qy, qz, qw])
        return np.array(tum_data).astype(np.float64)

    np.savetxt(
        fname=f"{filename}_tum.txt", X=_to_tum_format(poses, timestamps), fmt="%.4f"
    )


if __name__ == "__main__":
    argh.dispatch_command(main)
