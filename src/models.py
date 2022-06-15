from pydantic.dataclasses import dataclass
from pydantic import BaseModel

from typing import Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# todo add docs for all
# todo remove z.py, validate.py


Survey = Dict[str, Literal[1,2,3,4,5,6,7]]


# todo remove...?
class Request:
    user_id: str


# todo make not strict...
# todo fix
# @dataclass(extra='allow')
@dataclass
class Item:
    """
    Represents a generic item.
    """
    movielensId: str
    title: str
    genre: str


@dataclass
class Rating:
    """
    User's rating for an item. `rating` should be a number
    between 1 and 5 (both inclusive).
    """
    movielensId: str
    rating: Literal[1,2,3,4,5]


@dataclass
class EmotionInput:
    """
    User's input weight for the 8 dmotions. `weight` should be a percetenge decimal
    between 0 and 1 (both inclusive).
    """
    emotion: str
    weight: float
    
@dataclass
class EmoSliderInput:
    """
    User's input weight for the 8 dmotions. `weight` should be a percetenge decimal
    between 0 and 1 (both inclusive).
    """
    emotion: str
    switch: Literal["ignore", "diverse", "specified"]
    weight: float
    
    
@dataclass
class EmoButtonInput:
    """
    User's input weight for the 8 dmotions. `weight` should be a ordinal number
    among 0 1 2(both inclusive).
    """
    emotion: str
    weight: Literal["ignore", "diverse", "low", 'high']
    # corresponds to 0, 0, 0.3, 0.8 ???
    

@dataclass
class Recommendation:
    movielensId: str


@dataclass
class Preference:
    """
    Represents a predicted or actual preference. `categories`
    is a list of classes that an item belongs to.
    """
    movielensId: str
    # categories: List[Literal["CONTROVERSIAL", ""]] # todo more
    # categories: Literal["top_N", "diversified_by_latent_feature", "diversified_by_weighted_latent_feature", "diversified_by_emotions", "diversified_by_weighted_emotions"] # todo more
    categories: Literal["top_N", "diverse_N", "tuned_top_N_diverse_N"] # todo more

## for data viz
@dataclass
class LatentFeature:
    """
    Represents the latent feature values for visualization
    """
    movielensId: str
    feature1: float
    feature2: float
 
@dataclass
class EmotionalSignature:
    """
    Represents the latent feature values for visualization
    """
    movielensId: str
    # emotion1: float
    # emotion2: float 
    ############
    emotion1_name: str
    emotion1_val: float
    emotion2_name: str 
    emotion2_val: float 

@dataclass
class Event:
    """
    Represents an interaction with an item.
    """
    movielensId: str
    event_type: Literal["hover", "click"]
    duration: int
    enter_time: int
    exit_time: int