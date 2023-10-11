import pytest

import numpy as np
from fpfs.tasks import SimulationTask, ProcessSimulationTask


config_fname = "configuration_files/config_test.ini"
sim_task = SimulationTask(config_fname)


@pytest.mark.parametrize("ifield", (10, 24, 36))
def test_tasks(ifield):
    sim_task.clear(ifield)
    sim_task.run(ifield)

    meas_task = ProcessSimulationTask(config_fname)
    meas_task.clear(ifield)
    meas_task.run(ifield)

    src = meas_task.load_outcomes(ifield, "shape")
    for col in [0, 1]:
        np.testing.assert_allclose(
            np.sum(src["g2-0_rot0"][:, col]),
            np.sum(src["g2-0_rot1"][:, col]),
            rtol=3e-3,
        )
    for col in [2, 3]:
        np.testing.assert_allclose(
            np.sum(src["g2-0_rot0"][:, col]),
            -np.sum(src["g2-0_rot1"][:, col]),
            rtol=3e-3,
        )
    sim_task.clear(ifield)
    meas_task.clear(ifield)
    return


if __name__ == "__main__":
    test_tasks(100)
