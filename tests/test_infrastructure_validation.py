"""
Validation tests to ensure the testing infrastructure is properly set up.
"""
import sys
from pathlib import Path

import pytest


class TestInfrastructureValidation:
    """Tests to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_is_installed(self):
        """Test that pytest is properly installed."""
        assert "pytest" in sys.modules
    
    @pytest.mark.unit
    def test_project_structure_exists(self):
        """Test that the basic project structure exists."""
        workspace_root = Path(__file__).parent.parent
        
        # Check main directories
        assert workspace_root.exists()
        assert (workspace_root / "act").exists()
        assert (workspace_root / "teleop").exists()
        assert (workspace_root / "scripts").exists()
        assert (workspace_root / "tests").exists()
        
        # Check test structure
        assert (workspace_root / "tests" / "unit").exists()
        assert (workspace_root / "tests" / "integration").exists()
        assert (workspace_root / "tests" / "conftest.py").exists()
    
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, sample_config, mock_model):
        """Test that conftest fixtures are available."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test sample_config fixture
        assert isinstance(sample_config, dict)
        assert "model" in sample_config
        assert "training" in sample_config
        
        # Test mock_model fixture
        assert hasattr(mock_model, "forward")
        assert hasattr(mock_model, "parameters")
    
    @pytest.mark.unit
    def test_coverage_is_configured(self):
        """Test that coverage is properly configured."""
        try:
            import coverage
            assert coverage.__version__
        except ImportError:
            pytest.fail("Coverage module not installed")
    
    @pytest.mark.unit
    def test_markers_are_defined(self, request):
        """Test that custom markers are properly defined."""
        markers = request.config.getini("markers")
        markers_str = str(markers)
        assert "unit:" in markers_str
        assert "integration:" in markers_str
        assert "slow:" in markers_str
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Test that integration marker works correctly."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Test that slow marker works correctly."""
        # This test should be skipped by default unless --runslow is passed
        import time
        time.sleep(0.1)
        assert True
    
    @pytest.mark.unit
    def test_numpy_fixtures(self, sample_numpy_array):
        """Test numpy-related fixtures."""
        import numpy as np
        
        assert isinstance(sample_numpy_array, np.ndarray)
        assert sample_numpy_array.shape == (10, 20, 3)
        assert sample_numpy_array.dtype == np.float32
    
    @pytest.mark.unit
    def test_torch_fixtures(self, sample_torch_tensor, mock_dataset):
        """Test PyTorch-related fixtures."""
        import torch
        
        assert isinstance(sample_torch_tensor, torch.Tensor)
        assert sample_torch_tensor.shape == (8, 3, 224, 224)
        
        # Test mock dataset
        assert len(mock_dataset) == 100
        data, label = mock_dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, torch.Tensor)
    
    @pytest.mark.unit
    def test_file_fixtures(self, sample_yaml_config, sample_json_config):
        """Test file creation fixtures."""
        assert sample_yaml_config.exists()
        assert sample_yaml_config.suffix == ".yaml"
        
        assert sample_json_config.exists()
        assert sample_json_config.suffix == ".json"
    
    @pytest.mark.unit
    def test_environment_fixture(self, env_vars):
        """Test environment variable fixture."""
        import os
        
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("MODEL_PATH") == "/test/model/path"
        assert os.environ.get("DATA_DIR") == "/test/data"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
        (4, 16),
    ])
    def test_parametrize_works(self, value, expected):
        """Test that pytest parametrize decorator works."""
        assert value ** 2 == expected
    
    @pytest.mark.unit
    def test_mock_fixtures(self, mock_dynamixel_driver, mock_websocket):
        """Test mock fixtures are properly configured."""
        # Test Dynamixel driver mock
        assert mock_dynamixel_driver.connect() is True
        assert mock_dynamixel_driver.read_position() == 2048
        
        # Test WebSocket mock
        assert mock_websocket.closed is False
        data = mock_websocket.recv()
        assert isinstance(data, str)
        assert "test" in data


@pytest.mark.unit
class TestPoetryIntegration:
    """Tests to validate Poetry integration."""
    
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
    
    def test_pyproject_toml_has_poetry_config(self):
        """Test that pyproject.toml has Poetry configuration."""
        import toml
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        config = toml.load(pyproject_path)
        
        assert "tool" in config
        assert "poetry" in config["tool"]
        assert "dependencies" in config["tool"]["poetry"]
        assert "group" in config["tool"]["poetry"]
        assert "dev" in config["tool"]["poetry"]["group"]
        
    def test_test_scripts_are_defined(self):
        """Test that test scripts are defined in pyproject.toml."""
        import toml
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        config = toml.load(pyproject_path)
        
        assert "scripts" in config["tool"]["poetry"]
        assert "test" in config["tool"]["poetry"]["scripts"]
        assert "tests" in config["tool"]["poetry"]["scripts"]
        assert config["tool"]["poetry"]["scripts"]["test"] == "pytest"
        assert config["tool"]["poetry"]["scripts"]["tests"] == "pytest"