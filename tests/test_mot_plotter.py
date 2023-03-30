import pytest
import numpy as np
from fyp_model.bounding_box_plotter import BoundingBoxPlotter

class TestBoundingBoxPlotter:

    @pytest.fixture
    def img(self):
        return np.zeros((100, 100, 3))

    