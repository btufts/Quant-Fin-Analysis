from typing import List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


#
# 1) Base experiment model with fields common to both experiment types
#
class ExperimentBase(BaseModel):
    name: str
    type: str  # Will be discriminated to pick the right subclass
    logging: bool = False
    data: str
    symbol: str
    strategy: str
    cheat_on_open: bool = False
    cash: float = 100_000
    commission: float = 0.0


#
# 2) Specialized model for "optimize" experiment type
#    Now with dynamic 'parameters' that can be any key-value pairs
#
class OptimizeExperiment(ExperimentBase):
    type: Literal["optimize"]  # Must match exactly "optimize"

    opt_param: str = "cagr"
    opt_neighbors: int = 5
    # Accept any parameters as a dict
    parameters: Dict[str, Any] = Field(default_factory=dict)


#
# 3) Specialized model for "robustness" experiment type
#
class RobustnessExperiment(ExperimentBase):
    type: Literal["robustness"]  # Must match exactly "robustness"

    optimize_result: str
    tests: List[str]
    vsrandom_itrs: int = 100
    mcrandom_itrs: int = 100


#
# 4) A union of the two possible experiment models,
#    using a discriminator on the "type" field
#
Experiment = Union[OptimizeExperiment, RobustnessExperiment]


#
# 5) Top-level config model holding a list of experiments
#
class Config(BaseModel):
    experiments: List[Experiment]
