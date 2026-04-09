import sys
from os import path as osp

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.app import application_pipeline
except ModuleNotFoundError:
    from app import application_pipeline


if __name__ == '__main__':
    root_path = PROJECT_ROOT
    application_pipeline(root_path)
