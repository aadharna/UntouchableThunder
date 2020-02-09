import argparse
import yaml


def load_from_yaml(fpath):
    dic = yaml.load(open(fpath), Loader=yaml.FullLoader)
    return argparse.Namespace(**dic)
