from typing import Dict
import numpy as np
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    dataset_types = "UTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="datasets",
        datasets=datasets,
    )
    
    """============= Impletment your algo. ============="""
    from method.fan import FAN
    """============= Run your algo. ============="""
    # Specifying methods and training schemas
    
    training_schema = "naive"
    method = "FAN"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path="method/fan/config.toml"
    )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA, PointKthF1PA, PointAuprcPA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(mode="squeeze"),
            PointKthF1PA(k=5),
            PointAuprcPA()
        ]
    )

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
