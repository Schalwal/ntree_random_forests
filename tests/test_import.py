import pytest


def test_import():
    try:
        import ntrees_tuning
    except ImportError as e:
        pytest.fail(f"Importing your_package failed: {e}")
