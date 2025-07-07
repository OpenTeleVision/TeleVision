"""
Shared pytest fixtures and configuration for all tests.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator, Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 100,
            "device": "cpu",
        },
        "data": {
            "train_path": "/path/to/train",
            "val_path": "/path/to/val",
            "test_path": "/path/to/test",
        },
    }


@pytest.fixture
def sample_yaml_config(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a sample YAML configuration file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_json_config(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a sample JSON configuration file."""
    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Provide a sample numpy array for testing."""
    return np.random.randn(10, 20, 3).astype(np.float32)


@pytest.fixture
def sample_torch_tensor() -> torch.Tensor:
    """Provide a sample PyTorch tensor for testing."""
    return torch.randn(8, 3, 224, 224)


@pytest.fixture
def mock_model() -> MagicMock:
    """Provide a mock PyTorch model."""
    model = MagicMock()
    model.forward = MagicMock(return_value=torch.randn(8, 10))
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    model.train = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Provide a mock PyTorch dataset."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(
        return_value=(torch.randn(3, 224, 224), torch.tensor(1))
    )
    return dataset


@pytest.fixture
def sample_h5_data(temp_dir: Path) -> Path:
    """Create a sample HDF5 file with test data."""
    h5py = pytest.importorskip("h5py")
    h5_path = temp_dir / "test_data.h5"
    
    with h5py.File(h5_path, "w") as f:
        # Create sample datasets
        f.create_dataset("observations", data=np.random.randn(100, 10))
        f.create_dataset("actions", data=np.random.randn(100, 5))
        f.create_dataset("rewards", data=np.random.randn(100))
        
        # Create groups with nested data
        grp = f.create_group("metadata")
        grp.attrs["version"] = "1.0"
        grp.attrs["description"] = "Test HDF5 file"
        
    return h5_path


@pytest.fixture
def mock_robot_state() -> Dict[str, Any]:
    """Provide a mock robot state dictionary."""
    return {
        "joint_positions": np.random.randn(19).tolist(),
        "joint_velocities": np.random.randn(19).tolist(),
        "joint_torques": np.random.randn(19).tolist(),
        "gripper_position": [0.5, 0.5],
        "timestamp": 1234567890.123,
        "is_active": True,
    }


@pytest.fixture
def mock_camera_image() -> np.ndarray:
    """Provide a mock camera image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_dynamixel_driver() -> MagicMock:
    """Provide a mock Dynamixel driver."""
    driver = MagicMock()
    driver.connect = MagicMock(return_value=True)
    driver.disconnect = MagicMock()
    driver.read_position = MagicMock(return_value=2048)
    driver.write_position = MagicMock(return_value=True)
    driver.set_torque_enable = MagicMock(return_value=True)
    return driver


@pytest.fixture
def env_vars() -> Generator[Dict[str, str], None, None]:
    """Temporarily set environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        "TEST_VAR": "test_value",
        "MODEL_PATH": "/test/model/path",
        "DATA_DIR": "/test/data",
    }
    
    os.environ.update(test_env)
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_websocket() -> MagicMock:
    """Provide a mock WebSocket connection."""
    ws = MagicMock()
    ws.send = MagicMock()
    ws.recv = MagicMock(return_value='{"type": "test", "data": "test_data"}')
    ws.close = MagicMock()
    ws.closed = False
    return ws


@pytest.fixture
def sample_episode_data() -> Dict[str, Any]:
    """Provide sample episode data for testing."""
    return {
        "observations": [np.random.randn(10).tolist() for _ in range(50)],
        "actions": [np.random.randn(5).tolist() for _ in range(50)],
        "rewards": np.random.randn(50).tolist(),
        "done": [False] * 49 + [True],
        "info": {
            "episode_length": 50,
            "total_reward": 42.0,
            "success": True,
        },
    }


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset PyTorch random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # No cleanup needed


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb for tests that use it."""
    mock_wandb_module = MagicMock()
    mock_wandb_module.init = MagicMock()
    mock_wandb_module.log = MagicMock()
    mock_wandb_module.finish = MagicMock()
    mock_wandb_module.config = {}
    monkeypatch.setattr("wandb", mock_wandb_module)
    return mock_wandb_module


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip slow tests by default unless --runslow is passed
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip slow tests by default."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)