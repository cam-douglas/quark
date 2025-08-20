"""
Cloud-Based Training System for Small-Mind Agent Hub

This module enables continuous improvement of the agent hub through:
1. Feedback collection and analysis
2. Cloud-based model training
3. Performance metrics tracking
4. Automated optimization
"""

import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CloudTrainingManager:
    """Manages cloud-based training and improvement of the agent hub."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_db = config.get("feedback_db", "feedback_data.jsonl")
        self.metrics_db = config.get("metrics_db", "performance_metrics.jsonl")
        self.cloud_endpoint = config.get("cloud_endpoint", "https://api.smallmind.ai/training")
        self.api_key = config.get("api_key", "")
        self.training_interval = config.get("training_interval", 3600)  # 1 hour
        self.batch_size = config.get("batch_size", 1000)
        self.min_feedback_threshold = config.get("min_feedback_threshold", 100)
        
        # Training state
        self.last_training = None
        self.training_stats = {}
        self.model_versions = {}
        
        # Initialize databases
        self._init_databases()
    
    def _init_databases(self):
        """Initialize feedback and metrics databases."""
        for db_file in [self.feedback_db, self.metrics_db]:
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(db_file).exists():
                Path(db_file).touch()
    
    def collect_feedback(self, 
                        prompt: str, 
                        response: str, 
                        model_id: str, 
                        user_rating: int,
                        user_feedback: str = "",
                        execution_metrics: Dict[str, Any] = None) -> str:
        """
        Collect user feedback for training improvement.
        
        Args:
            prompt: User's original prompt
            response: Model's response
            model_id: ID of the model used
            user_rating: User rating (1-5)
            user_feedback: Optional text feedback
            execution_metrics: Performance metrics
            
        Returns:
            Feedback ID for tracking
        """
        feedback_id = str(uuid.uuid4())
        
        feedback_data = {
            "id": feedback_id,
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "response": response,
            "model_id": model_id,
            "user_rating": user_rating,
            "user_feedback": user_feedback,
            "execution_metrics": execution_metrics or {},
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
            "response_hash": hashlib.md5(response.encode()).hexdigest()
        }
        
        # Save to local database
        with open(self.feedback_db, "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        
        logger.info(f"Feedback collected: {feedback_id} (rating: {user_rating})")
        return feedback_id
    
    def collect_execution_metrics(self, 
                                 run_id: str, 
                                 model_id: str, 
                                 prompt: str,
                                 execution_time: float,
                                 resource_usage: Dict[str, Any],
                                 success: bool,
                                 error_message: str = "") -> str:
        """
        Collect execution metrics for performance analysis.
        
        Args:
            run_id: Unique run identifier
            model_id: Model used for execution
            prompt: User prompt
            execution_time: Time taken for execution
            resource_usage: CPU, memory, GPU usage
            success: Whether execution succeeded
            error_message: Error message if failed
            
        Returns:
            Metrics ID for tracking
        """
        metrics_id = str(uuid.uuid4())
        
        metrics_data = {
            "id": metrics_id,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "prompt": prompt,
            "execution_time": execution_time,
            "resource_usage": resource_usage,
            "success": success,
            "error_message": error_message,
            "prompt_length": len(prompt),
            "prompt_complexity": self._calculate_complexity(prompt)
        }
        
        # Save to metrics database
        with open(self.metrics_db, "a") as f:
            f.write(json.dumps(metrics_data) + "\n")
        
        logger.info(f"Metrics collected: {metrics_id} (time: {execution_time:.2f}s)")
        return metrics_id
    
    def _calculate_complexity(self, prompt: str) -> str:
        """Calculate prompt complexity for analysis."""
        word_count = len(prompt.split())
        has_code = any(keyword in prompt.lower() for keyword in ["code", "function", "class", "import"])
        has_technical = any(keyword in prompt.lower() for keyword in ["algorithm", "optimize", "performance", "scalability"])
        
        if has_code and has_technical and word_count > 50:
            return "high"
        elif (has_code or has_technical) and word_count > 30:
            return "medium"
        else:
            return "low"
    
    async def upload_to_cloud(self, data: List[Dict[str, Any]], data_type: str) -> bool:
        """
        Upload training data to cloud endpoint.
        
        Args:
            data: List of data records to upload
            data_type: Type of data (feedback, metrics, etc.)
            
        Returns:
            Success status
        """
        if not self.cloud_endpoint or not self.api_key:
            logger.warning("Cloud endpoint or API key not configured")
            return False
        
        try:
            import aiohttp
            
            payload = {
                "data_type": data_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
                "batch_size": len(data),
                "checksum": hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.cloud_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Successfully uploaded {len(data)} {data_type} records")
                        return True
                    else:
                        logger.error(f"Upload failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
            return False
    
    async def trigger_training(self) -> bool:
        """
        Trigger cloud-based training when sufficient data is available.
        
        Returns:
            Success status
        """
        # Check if we have enough feedback
        feedback_count = self._count_records(self.feedback_db)
        if feedback_count < self.min_feedback_threshold:
            logger.info(f"Insufficient feedback for training: {feedback_count}/{self.min_feedback_threshold}")
            return False
        
        try:
            # Prepare training data
            training_data = self._prepare_training_data()
            
            # Upload to cloud
            success = await self.upload_to_cloud(training_data, "training_batch")
            
            if success:
                self.last_training = datetime.utcnow()
                self._update_training_stats(training_data)
                logger.info("Training batch successfully uploaded to cloud")
                return True
            else:
                logger.error("Failed to upload training batch")
                return False
                
        except Exception as e:
            logger.error(f"Training trigger failed: {e}")
            return False
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from feedback and metrics."""
        training_data = []
        
        # Read feedback data
        feedback_records = self._read_records(self.feedback_db)
        metrics_records = self._read_records(self.metrics_db)
        
        # Combine and enrich data
        for feedback in feedback_records:
            # Find corresponding metrics
            metrics = next((m for m in metrics_records if m.get("run_id") == feedback.get("run_id")), {})
            
            training_example = {
                "prompt": feedback["prompt"],
                "expected_response": feedback["response"],
                "user_rating": feedback["user_rating"],
                "user_feedback": feedback["user_feedback"],
                "model_id": feedback["model_id"],
                "execution_metrics": metrics,
                "prompt_complexity": self._calculate_complexity(feedback["prompt"]),
                "response_quality": self._assess_response_quality(feedback["response"], feedback["user_rating"])
            }
            
            training_data.append(training_example)
        
        return training_data
    
    def _assess_response_quality(self, response: str, user_rating: int) -> Dict[str, Any]:
        """Assess response quality for training."""
        return {
            "length": len(response),
            "has_code": any(keyword in response.lower() for keyword in ["def ", "class ", "import ", "from "]),
            "has_structure": any(keyword in response.lower() for keyword in ["1.", "2.", "3.", "step", "first", "second"]),
            "user_rating": user_rating,
            "quality_score": user_rating / 5.0
        }
    
    def _count_records(self, db_file: str) -> int:
        """Count records in database file."""
        try:
            with open(db_file, "r") as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0
    
    def _read_records(self, db_file: str) -> List[Dict[str, Any]]:
        """Read records from database file."""
        records = []
        try:
            with open(db_file, "r") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records
    
    def _update_training_stats(self, training_data: List[Dict[str, Any]]):
        """Update training statistics."""
        self.training_stats = {
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "total_examples": len(training_data),
            "average_rating": sum(d["user_rating"] for d in training_data) / len(training_data),
            "complexity_distribution": self._calculate_complexity_distribution(training_data),
            "model_performance": self._calculate_model_performance(training_data)
        }
    
    def _calculate_complexity_distribution(self, training_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of prompt complexities."""
        distribution = {"low": 0, "medium": 0, "high": 0}
        for example in training_data:
            complexity = example.get("prompt_complexity", "low")
            distribution[complexity] += 1
        return distribution
    
    def _calculate_model_performance(self, training_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics per model."""
        model_stats = {}
        
        for example in training_data:
            model_id = example["model_id"]
            if model_id not in model_stats:
                model_stats[model_id] = {
                    "total_requests": 0,
                    "total_rating": 0,
                    "avg_rating": 0.0,
                    "complexity_breakdown": {"low": 0, "medium": 0, "high": 0}
                }
            
            stats = model_stats[model_id]
            stats["total_requests"] += 1
            stats["total_rating"] += example["user_rating"]
            stats["avg_rating"] = stats["total_rating"] / stats["total_requests"]
            
            complexity = example.get("prompt_complexity", "low")
            stats["complexity_breakdown"][complexity] += 1
        
        return model_stats
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and statistics."""
        return {
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "feedback_count": self._count_records(self.feedback_db),
            "metrics_count": self._count_records(self.metrics_db),
            "min_threshold": self.min_feedback_threshold,
            "ready_for_training": self._count_records(self.feedback_db) >= self.min_feedback_threshold,
            "training_stats": self.training_stats,
            "next_training_estimate": self._estimate_next_training()
        }
    
    def _estimate_next_training(self) -> Optional[str]:
        """Estimate when next training will occur."""
        if not self.last_training:
            return None
        
        feedback_count = self._count_records(self.feedback_db)
        if feedback_count >= self.min_feedback_threshold:
            return "Ready now"
        
        # Estimate based on current feedback collection rate
        # This is a simplified estimation
        remaining = self.min_feedback_threshold - feedback_count
        estimated_hours = remaining * 2  # Assume 2 hours per feedback on average
        
        next_training = self.last_training + timedelta(hours=estimated_hours)
        return next_training.isoformat()
    
    async def continuous_training_loop(self):
        """Continuous training loop for background operation."""
        while True:
            try:
                # Check if training is needed
                if self._count_records(self.feedback_db) >= self.min_feedback_threshold:
                    await self.trigger_training()
                
                # Wait for next iteration
                await asyncio.sleep(self.training_interval)
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

# Factory function
def create_training_manager(config: Dict[str, Any]) -> CloudTrainingManager:
    """Create a cloud training manager instance."""
    return CloudTrainingManager(config)

# Example configuration
EXAMPLE_CONFIG = {
    "feedback_db": "data/feedback.jsonl",
    "metrics_db": "data/metrics.jsonl",
    "cloud_endpoint": "https://api.smallmind.ai/training",
    "api_key": "your_api_key_here",
    "training_interval": 3600,  # 1 hour
    "batch_size": 1000,
    "min_feedback_threshold": 100
}
