import kornia_rs as K
import numpy as np


def test_icp_smoke():
    criteria = K.ICPConvergenceCriteria(max_iterations=100, tolerance=1e-6)
    assert criteria.max_iterations == 100
    assert criteria.tolerance == 1e-6

    result = K.ICPResult()
    assert result.rotation == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert result.translation == [0.0, 0.0, 0.0]
    assert result.num_iterations == 0
    assert result.rmse == 0.0


def test_icp_vanilla():
    # create a point cloud with 10 points
    source = np.ones((10, 3))
    target = np.ones((10, 3))

    # create a criteria
    criteria = K.ICPConvergenceCriteria(max_iterations=100, tolerance=1e-6)

    # create a initial rotation and translation
    initial_rot = np.eye(3)
    initial_trans = np.zeros(3)

    # run the icp algorithm
    result = K.icp_vanilla(source, target, initial_rot, initial_trans, criteria)

    # assert the result
    assert result.rotation == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert result.translation == [0.0, 0.0, 0.0]
    assert result.num_iterations == 2
    assert result.rmse == 0.0
