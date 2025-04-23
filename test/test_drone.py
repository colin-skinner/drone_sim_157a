import os, sys
sys.path.append(".")

import pytest

from drone import Drone

class TestConstructor:

    def test_dt_bad_input(self):

        # dt is less than 0
        with pytest.raises(ValueError) as e:
            Drone(-1)
        assert str(e.value) == "dt must be greater than 0"

        # dt is zero
        with pytest.raises(ValueError) as e:
            Drone(0)
        assert str(e.value) == "dt must be greater than 0"

        # state0 is not an ndarray
        with pytest.raises(ValueError) as e:
            Drone(1, 1)
        assert str(e.value) == "If state0 is input, must be an ndarray"

        