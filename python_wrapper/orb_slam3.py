import numpy as np
from python_wrapper import orb_slam_pybind as orb


class ORB_SLAM3:
    def __init__(self, vocabulary: str, config: str, sensor: str, vis: bool):

        """
        Initializes Orb Slam 3 pipeline
        Args:
            vocabulary: path to vocabulary
            config: path to config file
            sensor: sensor type case-insensitive (eg: "monocular", "stereo"...)
            vis: on/off visualisation

        """
        self.supported_sensors = {
            "MONOCULAR": orb._System.eSensor.MONOCULAR,
            "STEREO": orb._System.eSensor.STEREO,
            "RGBD": orb._System.eSensor.RGBD,
            "IMU_MONOCULAR": orb._System.eSensor.IMU_MONOCULAR,
            "IMU_STEREO": orb._System.eSensor.IMU_STEREO,
            "IMU_RGBD": orb._System.eSensor.IMU_RGBD,
        }
        eSensor = self.supported_sensors[sensor.upper()]
        # initialise orb slam here
        self.slam = orb._System(vocabulary, config, eSensor, vis)

    def TrackRGBD(
        self,
        image: np.ndarray,
        depthmap: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        # np.array([acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z,timestamp])(nx7)
        filename: str = "",
    ) -> None:
        assert isinstance(image, np.ndarray)
        assert isinstance(depthmap, np.ndarray)
        assert timestamp is not None
        if vImuMeas is not None:
            assert isinstance(vImuMeas, np.ndarray)
            return self.slam._TrackRGBD(image, depthmap, timestamp, vImuMeas, filename)
        if vImuMeas is None:
            return self.slam._TrackRGBD(image, depthmap, timestamp, filename=filename)

    def TrackMonocular(
        self,
        image: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        # np.array([acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z,timestamp])(nx7)
        filename: str = "",
    ) -> None:
        assert isinstance(image, np.ndarray)
        assert timestamp is not None
        if vImuMeas is not None:
            assert isinstance(vImuMeas, np.ndarray)
            return self.slam._TrackMonocular(image, timestamp, vImuMeas, filename)
        if vImuMeas is None:
            return self.slam._TrackMonocular(image, timestamp, filename=filename)

    def TrackStereo(
        self,
        imLeft: np.ndarray,
        imRight: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        # np.array([acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z,timestamp])(nx7)
        filename: str = "",
    ) -> None:
        assert isinstance(imLeft, np.ndarray)
        assert isinstance(imRight, np.ndarray)
        assert timestamp is not None
        if vImuMeas is not None:
            assert isinstance(vImuMeas, np.ndarray)
            return self.slam._TrackStereo(
                imLeft, imRight, timestamp, vImuMeas, filename
            )
        if vImuMeas is None:
            return self.slam._TrackStereo(imLeft, imRight, timestamp, filename=filename)

    def ActivateLocalizationMode(self):
        self.slam._ActivateLocalizationMode()

    def DeactivateLocalizationMode(self):
        self.slam._DeactivateLocalizationMode()

    def GetImageScale(self):
        return self.slam._GetImageScale()

    def MapChanged(self):
        return self.slam._MapChanged()

    def Reset(self):
        self.slam._Reset()

    def ResetActiveMap(self):
        self.slam._ResetActiveMap()

    def SaveKeyFrameTrajectoryEuroC(self, filename):
        self.slam._SaveKeyFrameTrajectoryEuroC(filename)

    def SaveTrajectoryEuroC(self, filename):
        self.slam._SaveTrajectoryEuroC(filename)

    def SaveKeyFrameTrajectoryTUM(self, filename):
        self.slam._SaveKeyFrameTrajectoryTUM(filename)

    def SaveTrajectoryKITTI(self, filename):
        self.slam._SaveTrajectoryKITTI(filename)

    def SaveTrajectoryTUM(self, filename):
        self.slam._SaveTrajectoryTUM(filename)

    def Shutdown(self):
        self.slam._Shutdown()

    def isFinished(self):
        return self.slam._isFinished()

    def isLost(self):
        return self.slam._isLost()

    def isShutDown(self):
        return self.slam._isShutDown()
