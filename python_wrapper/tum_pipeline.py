import argh
from tum import TUMDataset
from build.python_wrapper import orb_slam_pybind as orb

def main(data_source:str, config:str, vocabulary:str):
    dataset = TUMDataset(data_source)
    #initialise the system
    # orb.System(vocabulary, config, orb.System.eSensor.RGBD,True)


if __name__ == "__main__":
    argh.dispatch_command(main)

