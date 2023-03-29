import numpy as np
from python_wrapper import orb_slam_pybind as orb


class ORB_SLAM3:
    def __init__(self, vocabulary: str, config: str, sensor: str, vis: bool):

        self.supported_sensors = {
            "MONOCULAR": orb.System.eSensor.MONOCULAR,
            "STEREO": orb.System.eSensor.STEREO,
            "RGBD": orb.System.eSensor.RGBD,
            "IMU_MONOCULAR": orb.System.eSensor.IMU_MONOCULAR,
            "IMU_STEREO": orb.System.eSensor.IMU_STEREO,
            "IMU_RGBD": orb.System.eSensor.IMU_RGBD,
        }
        eSensor = self.supported_sensors[sensor.upper()]
        # initialise orb slam here
        self.slam = orb.System(vocabulary, config, eSensor, vis)

    def TrackRGBD(
        self,
        image: np.ndarray,
        depthmap: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        filename: str = "",
    ) -> None:
        assert isinstance(image, np.ndarray)
        assert isinstance(depthmap, np.ndarray)
        assert timestamp is not None
        if vImuMeas is not None:
            assert isinstance(vImuMeas, np.ndarray)
            return self.slam.TrackRGBD(image, depthmap, timestamp, vImuMeas, filename)
        if vImuMeas is None:
            return self.slam.TrackRGBD(image, depthmap, timestamp, filename=filename)

    def TrackMonocular(
        self,
        image: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        filename: str = "",
    ) -> None:
        assert isinstance(image, np.ndarray)
        assert timestamp is not None
        if vImuMeas is not None:
            assert isinstance(vImuMeas, np.ndarray)
            return self.slam.TrackMonocular(image, timestamp, vImuMeas, filename)
        if vImuMeas is None:
            return self.slam.TrackMonocular(image, timestamp, filename=filename)

    def TrackStereo(
        self,
        imLeft: np.ndarray,
        imRight: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        filename: str = "",
    ) -> None:
        assert isinstance(imLeft, np.ndarray)
        assert isinstance(imRight, np.ndarray)
        assert timestamp is not None
        if vImuMeas is not None:
            assert isinstance(vImuMeas, np.ndarray)
            return self.slam.TrackStereo(imLeft, imRight, timestamp, vImuMeas, filename)
        if vImuMeas is None:
            return self.slam.TrackStereo(imLeft, imRight, timestamp, filename=filename)

    def ActivateLocalizationMode(self):
        self.slam.ActivateLocalizationMode()

    def DeactivateLocalizationMode(self):
        self.slam.DeactivateLocalizationMode()

    def GetImageScale(self):
        return self.slam.GetImageScale()

    def MapChanged(self):
        return self.slam.MapChanged()

    def Reset(self):
        self.slam.Reset()

    def ResetActiveMap(self):
        self.slam.ResetActiveMap()

    def SaveKeyFrameTrajectoryEuroC(self, filename):
        self.slam.SaveKeyFrameTrajectoryEuRoC(filename)

    def SaveTrajectoryEuroC(self, filename):
        self.slam.SaveTrajectoryEuRoC(filename)

    def SaveKeyFrameTrajectoryTUM(self, filename):
        self.slam.SaveKeyFrameTrajectoryTUM(filename)

    def SaveTrajectoryKITTI(self, filename):
        self.slam.SaveTrajectoryKITTI(filename)

    def SaveTrajectoryTUM(self, filename):
        self.slam.SaveTrajectoryTUM(filename)

    def Shutdown(self):
        self.slam.Shutdown()

    def isFinished(self):
        return self.slam.isFinished()

    def isLost(self):
        return self.slam.isLost()

    def isShutDown(self):
        return self.slam.isShutDown()
