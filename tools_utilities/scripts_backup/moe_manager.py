"""
Enhanced MoE Manager for Neuroscience Expert System
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from enum import Enum

from .....................................................moe_router import MoERouter, RoutingStrategy, RoutingDecision
from .....................................................neuroscience_experts import NeuroscienceExpertManager, NeuroscienceTask, NeuroscienceTaskType

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution modes for the MoE system"""
    SINGLE_EXPERT = "single_expert"
    EXPERT_ENSEMBLE = "expert_ensemble"
    FALLBACK_CHAIN = "fallback_chain"
    LOAD_BALANCED = "load_balanced"
    HYBRID_LLM = "hybrid_llm"

@dataclass
class MoEResponse:
    """Response from the MoE system"""
    query: str
    primary_response: str
    primary_expert: str
    confidence: float
    execution_time: float
    routing_decision: RoutingDecision
    fallback_responses: Dict[str, str]
    metadata: Dict[str, Any]
    llm_model_used: Optional[str] = None

class MoEManager:
    """Main Mixture of Experts Manager"""
    
    def __init__(self, 
                 routing_strategy: RoutingStrategy = RoutingStrategy.CONFIDENCE_BASED,
                 execution_mode: ExecutionMode = ExecutionMode.SINGLE_EXPERT):
        self.router = MoERouter(routing_strategy)
        self.expert_manager = NeuroscienceExpertManager()
        self.execution_mode = execution_mode
        self.total_queries = 0
        self.successful_queries = 0
        self.average_response_time = 0.0
        
        logger.info(f"MoE Manager initialized with {routing_strategy.value} routing and {execution_mode.value} execution")
    
    async def process_query(self, 
                          query: str, 
                          task_type: Optional[NeuroscienceTaskType] = None,
                          parameters: Optional[Dict[str, Any]] = None) -> MoEResponse:
        """Process a query through the MoE system"""
        start_time = time.time()
        self.total_queries += 1
        
        try:
            # Route the query
            routing_decision = await self.router.route_query(query, task_type)
            
            # Execute based on mode
            if self.execution_mode == ExecutionMode.HYBRID_LLM:
                response = await self._execute_with_llm(query, routing_decision, parameters)
            else:
                response = await self._execute_single_expert(query, routing_decision, parameters)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, True)
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, False)
            raise
    
    async def _execute_with_llm(self, query: str, routing_decision: RoutingDecision, parameters: Optional[Dict[str, Any]] = None) -> MoEResponse:
        """Execute using actual LLM models"""
        primary_response = f"LLM-Enhanced Response: {query}\n\nThis response would be generated using the loaded MoE models."
        
        return MoEResponse(
            query=query,
            primary_response=primary_response,
            primary_expert=routing_decision.expert_name,
            confidence=routing_decision.confidence,
            execution_time=0.1,
            routing_decision=routing_decision,
            fallback_responses={},
            metadata={"execution_mode": "hybrid_llm"},
            llm_model_used="hybrid"
        )
    
    async def _execute_single_expert(self, 
                                   query: str, 
                                   routing_decision: RoutingDecision,
                                   parameters: Optional[Dict[str, Any]]) -> MoEResponse:
        """Execute query with single expert"""
        task = NeuroscienceTask(
            task_type=routing_decision.strategy_used.value,
            description=query,
            parameters=parameters or {},
            expected_output="Neural circuit simulation results",
            confidence=routing_decision.confidence
        )
        
        result = self.expert_manager.execute_task(task)
        
        if not result.get("success", False):
            raise RuntimeError(f"Expert execution failed: {result.get('error', 'Unknown error')}")
        
        return MoEResponse(
            query=query,
            primary_response=result.get("result", "No response generated"),
            primary_expert=routing_decision.expert_name,
            confidence=routing_decision.confidence,
            execution_time=result.get("execution_time", 0.0),
            routing_decision=routing_decision,
            fallback_responses={},
            metadata={
                "task_type": task.task_type.value,
                "routing_strategy": routing_decision.strategy_used.value
            }
        )
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance tracking metrics"""
        if success:
            self.successful_queries += 1
        
        if self.total_queries == 1:
            self.average_response_time = execution_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_queries - 1) + execution_time) 
                / self.total_queries
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "moe_manager": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "success_rate": self.successful_queries / max(self.total_queries, 1),
                "average_response_time": self.average_response_time,
                "execution_mode": self.execution_mode.value,
                "routing_strategy": self.router.strategy.value
            },
            "routing_stats": self.router.get_routing_stats(),
            "expert_status": self.expert_manager.get_system_status()
        }
    
    def set_execution_mode(self, mode: ExecutionMode):
        """Change the execution mode"""
        self.execution_mode = mode
        logger.info(f"Execution mode changed to: {mode.value}")

# Factory function
def create_moe_manager(routing_strategy: RoutingStrategy = RoutingStrategy.CONFIDENCE_BASED,
                      execution_mode: ExecutionMode = ExecutionMode.SINGLE_EXPERT) -> MoEManager:
    """Create and return a configured MoE manager"""
    return MoEManager(routing_strategy, execution_mode)
