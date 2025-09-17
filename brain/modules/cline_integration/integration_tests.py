"""
Cline Integration Tests - Comprehensive testing suite for Cline autonomous coding integration

This module provides complete testing coverage for the Cline integration with
Quark's brain architecture, including biological constraints, task delegation,
and autonomous coding capabilities.
"""

import asyncio
import json
import logging
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from brain.modules.cline_integration.cline_adapter import (
    ClineAdapter, CodingTask, TaskResult, TaskComplexity, ClineTaskType
)


class TestClineAdapter:
    """Test suite for ClineAdapter functionality"""
    
    @pytest.fixture
    def adapter(self):
        """Create ClineAdapter instance for testing"""
        return ClineAdapter()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample coding task for testing"""
        return CodingTask(
            description="Create a simple neural network module",
            task_type=ClineTaskType.CODE_GENERATION,
            complexity=TaskComplexity.MODERATE,
            files_involved=["brain/modules/test_neural_net.py"],
            biological_constraints=True
        )
    
    @pytest.mark.asyncio
    async def test_brain_context_loading(self, adapter):
        """Test brain context loading functionality"""
        context = await adapter.get_brain_context()
        
        assert isinstance(context, dict)
        assert "current_phase" in context
        assert "active_modules" in context
        assert "neural_architecture" in context
        assert "biological_constraints" in context
        assert "morphogen_status" in context
        
        # Verify biological constraints structure
        bio_constraints = context["biological_constraints"]
        assert bio_constraints["alphagenome_enabled"] is True
        assert "developmental_stage" in bio_constraints
        assert "cell_types_active" in bio_constraints
    
    @pytest.mark.asyncio
    async def test_biological_compliance_validation(self, adapter, sample_task):
        """Test biological constraint validation"""
        # Test valid task
        valid_result = await adapter._validate_biological_compliance(sample_task)
        assert valid_result is True
        
        # Test invalid task with prohibited patterns
        invalid_task = CodingTask(
            description="Create harmful neural network with negative_emotion patterns",
            task_type=ClineTaskType.CODE_GENERATION,
            complexity=TaskComplexity.MODERATE,
            files_involved=[],
            biological_constraints=True
        )
        
        invalid_result = await adapter._validate_biological_compliance(invalid_task)
        assert invalid_result is False
    
    @pytest.mark.asyncio
    async def test_task_complexity_assessment(self, adapter):
        """Test task complexity assessment logic"""
        # Simple task
        simple_desc = "Fix typo in comment"
        simple_complexity = adapter._assess_task_complexity(simple_desc)
        assert simple_complexity == TaskComplexity.SIMPLE
        
        # Moderate task
        moderate_desc = "Edit function to add new parameter"
        moderate_complexity = adapter._assess_task_complexity(moderate_desc)
        assert moderate_complexity == TaskComplexity.MODERATE
        
        # Complex task
        complex_desc = "Refactor multiple files in the system"
        complex_complexity = adapter._assess_task_complexity(complex_desc)
        assert complex_complexity == TaskComplexity.COMPLEX
        
        # Critical task
        critical_desc = "Modify brain architecture morphogen solver"
        critical_complexity = adapter._assess_task_complexity(critical_desc)
        assert critical_complexity == TaskComplexity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_task_delegation_logic(self, adapter, sample_task):
        """Test intelligent task delegation logic"""
        # Mock brain context
        with patch.object(adapter, 'get_brain_context', return_value={
            "current_phase": "Foundation Layer - SHH System Complete",
            "biological_constraints": {"developmental_stage": "neural_tube_closure"}
        }):
            
            # Test moderate complexity task - should delegate
            should_delegate = await adapter._should_delegate_to_cline(sample_task)
            assert should_delegate is True
            
            # Test critical task during foundation phase - should not delegate
            critical_task = CodingTask(
                description="Modify brain architecture",
                task_type=ClineTaskType.CODE_GENERATION,
                complexity=TaskComplexity.CRITICAL,
                files_involved=[]
            )
            
            should_not_delegate = await adapter._should_delegate_to_cline(critical_task)
            assert should_not_delegate is False
    
    @pytest.mark.asyncio
    async def test_post_execution_compliance(self, adapter):
        """Test post-execution biological compliance validation"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write valid content (under 300 lines)
            f.write("# Valid neural module\n" * 50)
            temp_path = f.name
        
        try:
            # Test valid result
            valid_result = TaskResult(
                success=True,
                output="Task completed",
                files_modified=[temp_path],
                commands_executed=[],
                biological_compliance=True
            )
            
            compliance = await adapter._validate_post_execution_compliance(valid_result)
            assert compliance is True
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
        
        # Test invalid result with oversized file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write invalid content (over 300 lines)
            f.write("# Invalid neural module\n" * 350)
            temp_path_invalid = f.name
        
        try:
            invalid_result = TaskResult(
                success=True,
                output="Task completed",
                files_modified=[temp_path_invalid],
                commands_executed=[],
                biological_compliance=True
            )
            
            compliance = await adapter._validate_post_execution_compliance(invalid_result)
            assert compliance is False
            
        finally:
            Path(temp_path_invalid).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_mcp_execution_simulation(self, mock_subprocess, adapter, sample_task):
        """Test MCP server execution simulation"""
        # Mock successful subprocess execution
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps({
                "output": "Task completed successfully",
                "files_modified": ["test_file.py"],
                "commands_executed": ["pytest"]
            }).encode(),
            b""
        )
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            brain_context = await adapter.get_brain_context()
            result = await adapter._execute_via_mcp(sample_task, brain_context)
            
            assert result.success is True
            assert "Task completed successfully" in result.output
            assert "test_file.py" in result.files_modified
    
    def test_configuration_loading(self, adapter):
        """Test configuration loading and validation"""
        assert adapter.workspace_path == Path("/Users/camdouglas/quark")
        assert adapter.autonomous_threshold == TaskComplexity.MODERATE
        assert adapter._cache_ttl == 300  # 5 minutes


class TestResourceManagerIntegration:
    """Test suite for ResourceManager Cline integration"""
    
    @pytest.fixture
    def mock_resource_manager(self):
        """Create mock ResourceManager with Cline integration"""
        from brain.architecture.neural_core.cognitive_systems.resource_management.manager_core import ResourceManager
        
        with patch.object(ResourceManager, '__init__', return_value=None):
            rm = ResourceManager()
            rm.cline_enabled = True
            rm.cline_adapter = Mock(spec=ClineAdapter)
            rm.logger = logging.getLogger('test')
            return rm
    
    @pytest.mark.asyncio
    async def test_complex_coding_task_handling(self, mock_resource_manager):
        """Test complex coding task handling via ResourceManager"""
        # Mock successful task result
        mock_task_result = TaskResult(
            success=True,
            output="Task completed via Cline",
            files_modified=["test_module.py"],
            commands_executed=["pytest tests/"],
            biological_compliance=True,
            execution_time=15.5
        )
        
        mock_resource_manager.cline_adapter.handle_complex_coding_task = AsyncMock(return_value=mock_task_result)
        
        result = await mock_resource_manager.handle_complex_coding_task(
            "Create advanced neural network module with biological constraints"
        )
        
        assert result["success"] is True
        assert result["biological_compliance"] is True
        assert result["execution_time"] == 15.5
        assert "test_module.py" in result["files_modified"]
    
    @pytest.mark.asyncio
    async def test_autonomous_code_generation(self, mock_resource_manager):
        """Test autonomous code generation functionality"""
        mock_result = TaskResult(
            success=True,
            output="Generated neural module",
            files_modified=["brain/modules/new_neural_module.py"],
            commands_executed=[],
            biological_compliance=True
        )
        
        mock_resource_manager.cline_adapter.autonomous_code_generation = AsyncMock(return_value=mock_result)
        mock_resource_manager.register_resource = Mock()
        
        result = await mock_resource_manager.autonomous_code_generation(
            "Generate morphogen gradient solver module"
        )
        
        assert result["success"] is True
        assert result["biological_compliance"] is True
        mock_resource_manager.register_resource.assert_called()
    
    @pytest.mark.asyncio
    async def test_neural_interface_testing(self, mock_resource_manager):
        """Test neural interface browser automation testing"""
        mock_result = TaskResult(
            success=True,
            output="Interface testing completed successfully",
            files_modified=[],
            commands_executed=[],
            biological_compliance=True
        )
        
        mock_resource_manager.cline_adapter.browser_automation_testing = AsyncMock(return_value=mock_result)
        
        result = await mock_resource_manager.test_neural_interface(
            "Test morphogen visualization interface",
            "http://localhost:3000/morphogen-viz"
        )
        
        assert result["success"] is True
        assert result["test_scenario"] == "Test morphogen visualization interface"
        assert result["app_url"] == "http://localhost:3000/morphogen-viz"


class TestBiologicalConstraints:
    """Test suite for biological constraint enforcement"""
    
    @pytest.mark.asyncio
    async def test_developmental_stage_validation(self):
        """Test developmental stage constraint validation"""
        adapter = ClineAdapter()
        
        # Mock brain context with neural tube closure stage
        with patch.object(adapter, 'get_brain_context', return_value={
            "biological_constraints": {
                "developmental_stage": "neural_tube_closure"
            }
        }):
            
            # Valid task for current stage
            valid_task = CodingTask(
                description="Implement SHH morphogen gradient",
                task_type=ClineTaskType.CODE_GENERATION,
                complexity=TaskComplexity.MODERATE,
                files_involved=[]
            )
            
            valid = await adapter._validate_biological_compliance(valid_task)
            assert valid is True
            
            # Invalid task for current stage (synaptogenesis during neural tube closure)
            invalid_task = CodingTask(
                description="Implement synaptogenesis module",
                task_type=ClineTaskType.CODE_GENERATION,
                complexity=TaskComplexity.MODERATE,
                files_involved=[]
            )
            
            invalid = await adapter._validate_biological_compliance(invalid_task)
            assert invalid is False
    
    def test_prohibited_pattern_detection(self):
        """Test detection of prohibited biological patterns"""
        adapter = ClineAdapter()
        
        prohibited_patterns = [
            "negative_emotion system",
            "harmful_pattern implementation", 
            "biological_harm module",
            "neural_damage function",
            "toxic_behavior class"
        ]
        
        for pattern in prohibited_patterns:
            task = CodingTask(
                description=f"Create {pattern} for testing",
                task_type=ClineTaskType.CODE_GENERATION,
                complexity=TaskComplexity.SIMPLE,
                files_involved=[]
            )
            
            # This should be detected as invalid
            # Note: Using asyncio.run for sync test
            result = asyncio.run(adapter._validate_biological_compliance(task))
            assert result is False, f"Pattern '{pattern}' should be prohibited"
    
    def test_file_size_compliance(self):
        """Test file size compliance (300 line limit)"""
        adapter = ClineAdapter()
        
        # Create oversized file content
        oversized_content = "\n".join([f"# Line {i}" for i in range(350)])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(oversized_content)
            temp_path = f.name
        
        try:
            result = TaskResult(
                success=True,
                output="Task completed",
                files_modified=[temp_path],
                commands_executed=[],
                biological_compliance=True
            )
            
            # This should fail compliance due to file size
            compliance = asyncio.run(adapter._validate_post_execution_compliance(result))
            assert compliance is False
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_morphogen_module_creation(self):
        """Test complete morphogen module creation scenario"""
        adapter = ClineAdapter()
        
        # Simulate creating a new morphogen module
        task_description = """
        Create a new morphogen module for WNT signaling pathway:
        - Implement WNT gradient calculation
        - Add biological parameter validation
        - Include cell fate specification rules
        - Follow architectural constraints (<300 lines)
        - Integrate with existing SHH system
        """
        
        with patch.object(adapter, '_execute_via_mcp') as mock_execute:
            mock_execute.return_value = TaskResult(
                success=True,
                output="WNT morphogen module created successfully",
                files_modified=[
                    "brain/modules/morphogen_solver/wnt_gradient_system.py",
                    "brain/modules/morphogen_solver/wnt_parameters.py"
                ],
                commands_executed=[
                    "pytest brain/modules/morphogen_solver/tests/",
                    "mypy --strict brain/modules/morphogen_solver/"
                ],
                biological_compliance=True,
                execution_time=45.2
            )
            
            result = await adapter.autonomous_code_generation(task_description)
            
            assert result.success is True
            assert result.biological_compliance is True
            assert len(result.files_modified) == 2
            assert "wnt_gradient_system.py" in result.files_modified[0]
            assert "pytest" in result.commands_executed[0]
    
    @pytest.mark.asyncio
    async def test_neural_interface_development_and_testing(self):
        """Test neural interface development with browser testing"""
        adapter = ClineAdapter()
        
        # First: Create the interface
        interface_task = "Create morphogen visualization web interface with real-time gradient display"
        
        with patch.object(adapter, '_execute_via_mcp') as mock_execute:
            mock_execute.return_value = TaskResult(
                success=True,
                output="Neural interface created with React and D3.js",
                files_modified=[
                    "web/src/components/MorphogenVisualizer.tsx",
                    "web/src/hooks/useMorphogenData.ts"
                ],
                commands_executed=["npm run build", "npm run dev"],
                biological_compliance=True
            )
            
            interface_result = await adapter.autonomous_code_generation(interface_task)
            assert interface_result.success is True
            
        # Then: Test the interface
        test_scenario = "Test morphogen gradient visualization with SHH data"
        
        with patch.object(adapter, '_execute_browser_automation') as mock_browser:
            mock_browser.return_value = TaskResult(
                success=True,
                output="Interface testing completed: All gradients render correctly",
                files_modified=[],
                commands_executed=[],
                biological_compliance=True
            )
            
            test_result = await adapter.browser_automation_testing(
                test_scenario, 
                "http://localhost:3000/morphogen-viz"
            )
            
            assert test_result.success is True
            assert "gradients render correctly" in test_result.output


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
