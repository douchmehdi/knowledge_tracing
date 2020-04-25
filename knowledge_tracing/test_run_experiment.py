import pytest
import pandas as pd
import numpy as np
from knowledge_tracing.run_experiments import encode_afm_bg, encode_afm_bgt, encode_pfa


@pytest.fixture
def basic_qmatrix_and_task_sessions():
    """ Returns as basic qmatrix Dataframe, and task_sessions Dataframe """
    qmatrix = pd.DataFrame([[1, 1, 0], [0, 1, 1]])
    sessions = [
        # student - task - start - solved
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 2, 1],
        [1, 1, 0, 1],
        [0, 0, 3, 1],
        [1, 0, 1, 0],
    ]
    # Student 0 solved Q0, then fail Q1, then solves Q1, then solves Q0 again
    # Student 1 solved Q1, then fail Q1 then fails Q0
    task_sessions = pd.DataFrame(
        sessions, columns=["student", "task", "start", "solved"]
    )
    return qmatrix, task_sessions


def test_encode_afm_bg(basic_qmatrix_and_task_sessions):

    qmatrix, task_sessions = basic_qmatrix_and_task_sessions
    X, y = encode_afm_bg(task_sessions, qmatrix)
    expected_X = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 2, 1],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 3, 0],
            [1, 1, 0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_almost_equal(X, expected_X)


def test_encode_afm_bgt(basic_qmatrix_and_task_sessions):
    qmatrix, task_sessions = basic_qmatrix_and_task_sessions
    X, y = encode_afm_bgt(task_sessions, qmatrix)
    expected_X = np.array(
        [
            [1, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 2, 1, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 3, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 1, 0],
        ]
    )
    np.testing.assert_array_almost_equal(X, expected_X)


def test_encode_pfa(basic_qmatrix_and_task_sessions):
    qmatrix, task_sessions = basic_qmatrix_and_task_sessions
    X, y = encode_pfa(task_sessions, qmatrix)
    expected_X = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 2, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_almost_equal(X, expected_X)
