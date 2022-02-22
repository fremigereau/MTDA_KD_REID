from __future__ import absolute_import

from .softmax import ImageSoftmaxEngine
from .triplet import ImageTripletEngine
from .mmd import ImageMmdEngine
from .adv_MTDA import AdvMTDAEngine
from .margin_partial_L2_per_batch import MarginMTDAEnginePerBatch
from .margin_partial_L2_one_by_one import MarginMTDAEngineOnebyOne
from .visual_engine import Visual_Engine
from .Shift_measure_engine import DomainShiftEngine