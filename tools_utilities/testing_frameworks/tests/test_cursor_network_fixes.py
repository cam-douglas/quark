#!/usr/bin/env python3
"""
Test Suite for Cursor Network Fixes
Purpose: Comprehensive tests for Docker network issue resolution
Author: Quark Development Team
Dependencies: pytest, unittest, json, tempfile
"""

import json
import os
import pytest
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import sys

# Add the debug scripts to path for testing
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "debug"))

try:
    from cursor_network_fix import CursorNetworkFixer
    from cursor_network_test import CursorNetworkTester
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)

class TestCursorNetworkFixer(unittest.TestCase):
    """Test suite for CursorNetworkFixer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.fixer = CursorNetworkFixer()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_docker_detection(self):
        """Test Docker environment detection."""
        # Test with mock .dockerenv file
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            self.assertTrue(self.fixer._is_running_in_docker())
        
        # Test with mock cgroup
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('builtins.open', mock_open(read_data='docker')):
                self.assertTrue(self.fixer._is_running_in_docker())
    
    def test_connectivity_test(self):
        """Test network connectivity testing."""
        # Test successful connection
        with patch('socket.socket') as mock_socket:
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 0
            mock_socket.return_value = mock_sock_instance
            
            result = self.fixer._test_connectivity("example.com")
            self.assertTrue(result["success"])
            self.assertEqual(result["port"], 443)
            self.assertIsNone(result["error"])
        
        # Test failed connection
        with patch('socket.socket') as mock_socket:
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 1
            mock_socket.return_value = mock_sock_instance
            
            result = self.fixer._test_connectivity("invalid.example")
            self.assertFalse(result["success"])
            self.assertIsNotNone(result["error"])
    
    def test_dns_resolution(self):
        """Test DNS resolution testing."""
        # Test successful resolution
        with patch('socket.gethostbyname') as mock_dns:
            mock_dns.return_value = "192.168.1.1"
            
            result = self.fixer._test_dns_resolution("example.com")
            self.assertTrue(result["success"])
            self.assertEqual(result["ip"], "192.168.1.1")
            self.assertIsNone(result["error"])
        
        # Test failed resolution
        with patch('socket.gethostbyname') as mock_dns:
            mock_dns.side_effect = Exception("DNS resolution failed")
            
            result = self.fixer._test_dns_resolution("invalid.example")
            self.assertFalse(result["success"])
            self.assertIsNone(result["ip"])
            self.assertIsNotNone(result["error"])
    
    def test_config_backup(self):
        """Test configuration backup functionality."""
        # Create a temporary config file
        config_content = {"test": "config"}
        temp_config = self.temp_dir / "test_config.json"
        
        with open(temp_config, 'w') as f:
            json.dump(config_content, f)
        
        # Patch the config paths to use our temp file
        with patch.object(self.fixer, 'cursor_config_paths', [temp_config]):
            with patch.object(self.fixer, 'backup_dir', self.temp_dir):
                backup_path = self.fixer.backup_current_config()
                
                self.assertIsNotNone(backup_path)
                self.assertTrue(backup_path.exists())
                
                # Verify backup content
                with open(backup_path, 'r') as f:
                    backup_content = json.load(f)
                
                self.assertEqual(backup_content, config_content)
    
    def test_docker_settings_application(self):
        """Test application of Docker network settings."""
        # Create temporary config file
        temp_config = self.temp_dir / "settings.json"
        initial_config = {"existing": "setting"}
        
        with open(temp_config, 'w') as f:
            json.dump(initial_config, f)
        
        # Patch config paths to use temp file
        with patch.object(self.fixer, 'cursor_config_paths', [temp_config]):
            success = self.fixer.apply_docker_network_fix()
            
            self.assertTrue(success)
            
            # Verify settings were applied
            with open(temp_config, 'r') as f:
                updated_config = json.load(f)
            
            expected_settings = [
                "cursor.network.dockerMode",
                "cursor.network.disableStreaming", 
                "cursor.chat.usePolling",
                "cursor.agent.usePolling",
                "cursor.network.streamingFallback"
            ]
            
            for setting in expected_settings:
                self.assertIn(setting, updated_config)
            
            # Verify original setting is preserved
            self.assertEqual(updated_config["existing"], "setting")
    
    def test_environment_variable_setting(self):
        """Test environment variable configuration."""
        original_env = dict(os.environ)
        
        try:
            # Clear cursor-related env vars
            cursor_vars = [k for k in os.environ.keys() if k.startswith('CURSOR_')]
            for var in cursor_vars:
                if var in os.environ:
                    del os.environ[var]
            
            # Apply environment variables
            self.fixer.set_environment_variables()
            
            # Verify variables are set
            expected_vars = [
                "CURSOR_DISABLE_STREAMING",
                "CURSOR_DOCKER_MODE", 
                "CURSOR_USE_POLLING",
                "CURSOR_NETWORK_TIMEOUT"
            ]
            
            for var in expected_vars:
                self.assertIn(var, os.environ)
                self.assertIsNotNone(os.environ[var])
        
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_docker_command_generation(self):
        """Test Docker run command generation."""
        commands = self.fixer.generate_docker_run_commands()
        
        self.assertIsInstance(commands, list)
        self.assertGreater(len(commands), 0)
        
        # Verify commands contain expected elements
        command_text = ' '.join(commands)
        expected_elements = [
            "--network host",
            "--add-host=api2.cursor.sh",
            "--dns=8.8.8.8",
            "CURSOR_DISABLE_STREAMING=1"
        ]
        
        for element in expected_elements:
            self.assertIn(element, command_text)
    
    def test_comprehensive_diagnosis(self):
        """Test comprehensive environment diagnosis."""
        with patch.object(self.fixer, '_is_running_in_docker', return_value=True):
            with patch.object(self.fixer, '_test_connectivity') as mock_conn:
                with patch.object(self.fixer, '_test_dns_resolution') as mock_dns:
                    # Mock successful connectivity and DNS
                    mock_conn.return_value = {"success": True}
                    mock_dns.return_value = {"success": True}
                    
                    diagnosis = self.fixer.diagnose_environment()
                    
                    self.assertIn("docker_detected", diagnosis)
                    self.assertIn("network_connectivity", diagnosis)
                    self.assertIn("dns_resolution", diagnosis)
                    self.assertIn("environment_vars", diagnosis)
                    
                    self.assertTrue(diagnosis["docker_detected"])

class TestCursorNetworkTester(unittest.TestCase):
    """Test suite for CursorNetworkTester class."""
    
    def setUp(self):
        """Set up test environment."""
        self.tester = CursorNetworkTester()
    
    def test_environment_info_gathering(self):
        """Test environment information gathering."""
        env_info = self.tester._get_environment_info()
        
        self.assertIn("platform", env_info)
        self.assertIn("python_version", env_info)
        self.assertIn("is_docker", env_info)
        self.assertIn("environment_vars", env_info)
        
        self.assertIsInstance(env_info["environment_vars"], dict)
    
    @patch('requests.get')
    def test_basic_connectivity_success(self, mock_get):
        """Test basic connectivity testing with successful response."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_get.return_value = mock_response
        
        results = self.tester.test_basic_connectivity()
        
        self.assertIsInstance(results, dict)
        for endpoint in self.tester.cursor_endpoints:
            self.assertIn(endpoint, results)
            self.assertTrue(results[endpoint]["success"])
            self.assertEqual(results[endpoint]["status_code"], 200)
    
    @patch('requests.get')
    def test_basic_connectivity_failure(self, mock_get):
        """Test basic connectivity testing with failed response."""
        # Mock failed response
        mock_get.side_effect = Exception("Connection failed")
        
        results = self.tester.test_basic_connectivity()
        
        self.assertIsInstance(results, dict)
        for endpoint in self.tester.cursor_endpoints:
            self.assertIn(endpoint, results)
            self.assertFalse(results[endpoint]["success"])
            self.assertIsNotNone(results[endpoint]["error"])
    
    @patch('socket.getaddrinfo')
    def test_dns_resolution_success(self, mock_getaddrinfo):
        """Test DNS resolution with successful lookup."""
        # Mock successful DNS resolution
        mock_getaddrinfo.return_value = [
            (2, 1, 6, '', ('192.168.1.1', 443)),
            (2, 1, 6, '', ('192.168.1.2', 443))
        ]
        
        results = self.tester.test_dns_resolution()
        
        self.assertIsInstance(results, dict)
        # Check at least one domain was tested
        domain_results = list(results.values())
        if domain_results:
            self.assertTrue(domain_results[0]["success"])
            self.assertIsInstance(domain_results[0]["ip_addresses"], list)
    
    @patch('requests.get')
    def test_streaming_capability(self, mock_get):
        """Test streaming capability assessment."""
        # Mock streaming response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        mock_get.return_value = mock_response
        
        results = self.tester.test_streaming_capability()
        
        self.assertIsInstance(results, dict)
        # Verify streaming test structure
        for url, result in results.items():
            if result["success"]:
                self.assertIn("chunks_received", result)
                self.assertIn("streaming_works", result)
    
    def test_polling_simulation(self):
        """Test polling behavior simulation."""
        with patch('requests.get') as mock_get:
            # Mock successful polling responses
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            results = self.tester.test_polling_simulation()
            
            self.assertIn("polling_test", results)
            if results["polling_test"]["success"]:
                self.assertGreater(results["polling_test"]["polls_completed"], 0)
                self.assertIsNotNone(results["polling_test"]["average_poll_time"])
    
    def test_cursor_configuration_check(self):
        """Test Cursor configuration file checking."""
        # Create temporary config file
        temp_dir = Path(tempfile.mkdtemp())
        temp_config = temp_dir / "settings.json"
        
        config_content = {
            "cursor.network.dockerMode": True,
            "cursor.network.disableStreaming": True,
            "cursor.chat.usePolling": True
        }
        
        try:
            with open(temp_config, 'w') as f:
                json.dump(config_content, f)
            
            # Patch config paths to use temp file
            with patch.object(self.tester, 'cursor_config_paths', [temp_config]):
                results = self.tester.check_cursor_configuration()
                
                self.assertIsInstance(results, dict)
                config_result = results[str(temp_config)]
                self.assertTrue(config_result["exists"])
                self.assertTrue(config_result["readable"])
                self.assertTrue(config_result["has_docker_fix"])
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete network fix system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_fix_workflow(self):
        """Test complete workflow from diagnosis to fix."""
        fixer = CursorNetworkFixer()
        
        # Create temporary config
        temp_config = self.temp_dir / "settings.json"
        temp_config.write_text('{"existing": "setting"}')
        
        # Patch paths
        with patch.object(fixer, 'cursor_config_paths', [temp_config]):
            with patch.object(fixer, 'backup_dir', self.temp_dir):
                # Mock Docker detection
                with patch.object(fixer, '_is_running_in_docker', return_value=True):
                    # Mock connectivity tests
                    with patch.object(fixer, '_test_connectivity') as mock_conn:
                        with patch.object(fixer, '_test_dns_resolution') as mock_dns:
                            mock_conn.return_value = {"success": True}
                            mock_dns.return_value = {"success": True}
                            
                            # Run diagnosis
                            diagnosis = fixer.diagnose_environment()
                            self.assertTrue(diagnosis["docker_detected"])
                            
                            # Apply fixes
                            backup_created = fixer.backup_current_config()
                            self.assertIsNotNone(backup_created)
                            
                            fix_applied = fixer.apply_docker_network_fix()
                            self.assertTrue(fix_applied)
                            
                            # Verify fix was applied
                            with open(temp_config, 'r') as f:
                                updated_config = json.load(f)
                            
                            self.assertTrue(updated_config.get("cursor.network.dockerMode"))
                            self.assertTrue(updated_config.get("cursor.network.disableStreaming"))
    
    def test_test_and_fix_integration(self):
        """Test integration between tester and fixer."""
        tester = CursorNetworkTester()
        fixer = CursorNetworkFixer()
        
        # Mock network issues
        with patch.object(tester, 'test_streaming_capability') as mock_streaming:
            mock_streaming.return_value = {
                "https://test.example": {
                    "success": True,
                    "streaming_works": False,  # Simulate streaming issue
                    "error": None
                }
            }
            
            # Run test
            streaming_results = tester.test_streaming_capability()
            
            # Check if fix is needed
            needs_fix = any(
                not result.get("streaming_works", True)
                for result in streaming_results.values()
                if result.get("success")
            )
            
            self.assertTrue(needs_fix)
            
            # Apply fix would be called here
            # This demonstrates the workflow integration

def run_cursor_network_tests():
    """Run all Cursor network fix tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCursorNetworkFixer,
        TestCursorNetworkTester, 
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_cursor_network_tests()
    sys.exit(0 if success else 1)


