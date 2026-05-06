# DO NOT CHANGE THIS, if you change this old models will become incompatible
# Compatibility problem: the ARX bundle was saved with ReducedRankRidge
# at fusion.training.train_joint_arx_v3. This file re-exports it from
# arx_model.py so joblib can deserialise the bundle when this folder
# is placed as the fusion/ package.
from .arx_model import ReducedRankRidge  # noqa: F401
