import unittest
import numpy as np

import myutils as my


class TestMyUtils(unittest.TestCase):
    def test_row_reduce_py(self):
        m = np.array([
            [3., -3, 0],
            [1, 2, 3],
            [7, -5, 2],
            [3, -1, 2]])
        expected = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]])
        result = my.row_reduce_py(m)
        self.assertTrue(np.allclose(result, expected, atol=my.tolerance))

        m = np.array([
            [1., 2, 1],
            [-2, -3, 1],
            [3, 5, 0]])
        expected = np.array([
            [1, 0, -5],
            [0, 1, 3],
            [0, 0, 0]])
        result = my.row_reduce_py(m)
        self.assertTrue(np.allclose(result, expected, atol=my.tolerance))

        m = np.array([
            [1., 0, 2, 0],
            [1, 1, 0, 0],
            [1, 2, 0, 1],
            [1, 1, 1, 1]])
        expected = np.identity(4)
        result = my.row_reduce_py(m)
        self.assertTrue(np.allclose(result, expected, atol=my.tolerance))

    def test_row_reduce_c(self):
        m = np.array([
            [3., -3, 0],
            [1, 2, 3],
            [7, -5, 2],
            [3, -1, 2]])
        expected = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]])
        result = my.row_reduce_c(m)
        self.assertTrue(np.allclose(result, expected, atol=my.tolerance))

        m = np.array([
            [1., 2, 1],
            [-2, -3, 1],
            [3, 5, 0]])
        expected = np.array([
            [1, 0, -5],
            [0, 1, 3],
            [0, 0, 0]])
        result = my.row_reduce_c(m)
        self.assertTrue(np.allclose(result, expected, atol=my.tolerance))

        m = np.array([
            [1., 0, 2, 0],
            [1, 1, 0, 0],
            [1, 2, 0, 1],
            [1, 1, 1, 1]])
        expected = np.identity(4)
        result = my.row_reduce_c(m)
        self.assertTrue(np.allclose(result, expected, atol=my.tolerance))
