# checkpoint_manager.py
"""
Model checkpoint management system for grid trading
Handles model versioning, saving, loading, and rollback

Author: Grid Trading System
Date: 2024
"""

import asyncio
import time
import logging
import json
import pickle
import hashlib
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import joblib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHECKPOINT_DIR = "checkpoints"
MAX_CHECKPOINTS_PER_MODEL = 10
CHECKPOINT_FORMAT_VERSION = "1.0.0"
METADATA_FILENAME = "checkpoint_metadata.json"
STATE_FILENAME = "model_state.pkl"
PERFORMANCE_FILENAME = "performance_metrics.json"
VALIDATION_FILENAME = "validation_results.json"


class CheckpointStatus(Enum):
    """Checkpoint status types"""
    ACTIVE = "active"          # Currently active checkpoint
    VALIDATED = "validated"    # Passed validation
    FAILED = "failed"         # Failed validation
    ARCHIVED = "archived"     # Archived for history
    CORRUPTED = "corrupted"   # Corrupted data


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    model_name: str
    version: str
    timestamp: float
    status: CheckpointStatus
    performance_metrics: Dict[str, float]
    training_info: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]] = None
    overfitting_metrics: Optional[Dict[str, float]] = None
    file_hashes: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'model_name': self.model_name,
            'version': self.version,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'performance_metrics': self.performance_metrics,
            'training_info': self.training_info,
            'validation_results': self.validation_results,
            'overfitting_metrics': self.overfitting_metrics,
            'file_hashes': self.file_hashes,
            'tags': self.tags,
            'notes': self.notes
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary"""
        data['status'] = CheckpointStatus(data['status'])
        return cls(**data)


@dataclass
class CheckpointValidation:
    """Validation results for a checkpoint"""
    is_valid: bool
    validation_score: float
    checks_passed: List[str]
    checks_failed: List[str]
    performance_comparison: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class BaseCheckpointable(ABC):
    """Base class for checkpointable components"""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing"""
        pass
        
    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint"""
        pass
        
    @abstractmethod
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """Get metadata for checkpoint"""
        pass


class CheckpointManager:
    """Manages model checkpoints and versioning"""
    
    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint registry
        self.checkpoints: Dict[str, List[CheckpointMetadata]] = defaultdict(list)
        self.active_checkpoints: Dict[str, str] = {}  # model_name -> checkpoint_id
        
        # Performance tracking
        self.checkpoint_history = deque(maxlen=100)
        self.rollback_history = deque(maxlen=50)
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Load existing checkpoints
        asyncio.create_task(self._load_checkpoint_registry())
        
    async def save_checkpoint(self,
                            model_name: str,
                            component: BaseCheckpointable,
                            performance_metrics: Dict[str, float],
                            force: bool = False) -> str:
        """Save model checkpoint"""
        async with self._lock:
            try:
                # Generate checkpoint ID
                checkpoint_id = self._generate_checkpoint_id(model_name)
                
                # Create checkpoint directory
                checkpoint_path = self.checkpoint_dir / model_name / checkpoint_id
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Get component state
                state = component.get_state()
                
                # Save state
                state_path = checkpoint_path / STATE_FILENAME
                await self._save_state(state, state_path)
                
                # Get and save metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    model_name=model_name,
                    version=CHECKPOINT_FORMAT_VERSION,
                    timestamp=time.time(),
                    status=CheckpointStatus.ACTIVE,
                    performance_metrics=performance_metrics,
                    training_info=component.get_checkpoint_metadata(),
                    file_hashes={
                        STATE_FILENAME: self._calculate_file_hash(state_path)
                    }
                )
                
                # Add overfitting metrics if available
                if hasattr(component, 'get_overfitting_metrics'):
                    metadata.overfitting_metrics = component.get_overfitting_metrics()
                
                # Save metadata
                metadata_path = checkpoint_path / METADATA_FILENAME
                await self._save_metadata(metadata, metadata_path)
                
                # Save performance metrics
                perf_path = checkpoint_path / PERFORMANCE_FILENAME
                await self._save_json(performance_metrics, perf_path)
                
                # Update registry
                self.checkpoints[model_name].append(metadata)
                self.active_checkpoints[model_name] = checkpoint_id
                
                # Cleanup old checkpoints
                await self._cleanup_old_checkpoints(model_name)
                
                # Log checkpoint
                self.checkpoint_history.append({
                    'checkpoint_id': checkpoint_id,
                    'model_name': model_name,
                    'timestamp': time.time(),
                    'action': 'save'
                })
                
                logger.info(f"Saved checkpoint {checkpoint_id} for {model_name}")
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                raise
                
    async def load_checkpoint(self,
                            model_name: str,
                            component: BaseCheckpointable,
                            checkpoint_id: Optional[str] = None) -> bool:
        """Load model checkpoint"""
        async with self._lock:
            try:
                # Use latest checkpoint if not specified
                if checkpoint_id is None:
                    checkpoint_id = self.active_checkpoints.get(model_name)
                    if not checkpoint_id:
                        logger.warning(f"No active checkpoint for {model_name}")
                        return False
                        
                # Get checkpoint path
                checkpoint_path = self.checkpoint_dir / model_name / checkpoint_id
                if not checkpoint_path.exists():
                    logger.error(f"Checkpoint path not found: {checkpoint_path}")
                    return False
                    
                # Verify checkpoint integrity
                if not await self._verify_checkpoint_integrity(checkpoint_path):
                    logger.error(f"Checkpoint integrity check failed: {checkpoint_id}")
                    return False
                    
                # Load state
                state_path = checkpoint_path / STATE_FILENAME
                state = await self._load_state(state_path)
                
                # Apply state to component
                component.load_state(state)
                
                # Log loading
                self.checkpoint_history.append({
                    'checkpoint_id': checkpoint_id,
                    'model_name': model_name,
                    'timestamp': time.time(),
                    'action': 'load'
                })
                
                logger.info(f"Loaded checkpoint {checkpoint_id} for {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False
                
    async def rollback_to_checkpoint(self,
                                   model_name: str,
                                   component: BaseCheckpointable,
                                   checkpoint_id: str,
                                   reason: str = "") -> bool:
        """Rollback to specific checkpoint"""
        async with self._lock:
            try:
                # Validate checkpoint exists
                checkpoint_metadata = await self._get_checkpoint_metadata(
                    model_name, 
                    checkpoint_id
                )
                
                if not checkpoint_metadata:
                    logger.error(f"Checkpoint not found: {checkpoint_id}")
                    return False
                    
                # Save current state as backup
                current_checkpoint = await self.save_checkpoint(
                    f"{model_name}_rollback_backup",
                    component,
                    {},
                    force=True
                )
                
                # Load target checkpoint
                success = await self.load_checkpoint(
                    model_name,
                    component,
                    checkpoint_id
                )
                
                if success:
                    # Update active checkpoint
                    self.active_checkpoints[model_name] = checkpoint_id
                    
                    # Log rollback
                    self.rollback_history.append({
                        'model_name': model_name,
                        'from_checkpoint': current_checkpoint,
                        'to_checkpoint': checkpoint_id,
                        'reason': reason,
                        'timestamp': time.time()
                    })
                    
                    logger.info(f"Rolled back {model_name} to checkpoint {checkpoint_id}")
                    
                return success
                
            except Exception as e:
                logger.error(f"Failed to rollback: {e}")
                return False
                
    async def validate_checkpoint(self,
                                model_name: str,
                                checkpoint_id: str,
                                validation_data: Optional[Any] = None) -> CheckpointValidation:
        """Validate a checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / model_name / checkpoint_id
            
            validation = CheckpointValidation(
                is_valid=True,
                validation_score=1.0,
                checks_passed=[],
                checks_failed=[],
                performance_comparison={}
            )
            
            # Check 1: File integrity
            if await self._verify_checkpoint_integrity(checkpoint_path):
                validation.checks_passed.append("file_integrity")
            else:
                validation.checks_failed.append("file_integrity")
                validation.is_valid = False
                validation.validation_score *= 0.0
                
            # Check 2: Metadata consistency
            metadata = await self._load_checkpoint_metadata(checkpoint_path)
            if metadata and self._validate_metadata(metadata):
                validation.checks_passed.append("metadata_consistency")
            else:
                validation.checks_failed.append("metadata_consistency")
                validation.validation_score *= 0.8
                
            # Check 3: Performance metrics
            if metadata and metadata.performance_metrics:
                perf_score = self._evaluate_performance_metrics(metadata.performance_metrics)
                validation.performance_comparison = {
                    'win_rate': metadata.performance_metrics.get('win_rate', 0),
                    'profit_factor': metadata.performance_metrics.get('profit_factor', 0),
                    'performance_score': perf_score
                }
                
                if perf_score > 0.5:
                    validation.checks_passed.append("performance_metrics")
                else:
                    validation.checks_failed.append("performance_metrics")
                    validation.validation_score *= perf_score
                    
            # Check 4: Overfitting indicators
            if metadata and metadata.overfitting_metrics:
                overfitting_score = self._evaluate_overfitting_metrics(
                    metadata.overfitting_metrics
                )
                
                if overfitting_score > 0.7:
                    validation.checks_passed.append("overfitting_check")
                else:
                    validation.checks_failed.append("overfitting_check")
                    validation.validation_score *= overfitting_score
                    
            # Save validation results
            if checkpoint_path.exists():
                validation_path = checkpoint_path / VALIDATION_FILENAME
                await self._save_json(validation.__dict__, validation_path)
                
            return validation
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return CheckpointValidation(
                is_valid=False,
                validation_score=0.0,
                checks_passed=[],
                checks_failed=["exception"],
                performance_comparison={}
            )
            
    async def get_checkpoint_history(self, 
                                   model_name: Optional[str] = None,
                                   limit: int = 50) -> List[Dict[str, Any]]:
        """Get checkpoint history"""
        history = []
        
        if model_name:
            checkpoints = self.checkpoints.get(model_name, [])
        else:
            checkpoints = []
            for model_checkpoints in self.checkpoints.values():
                checkpoints.extend(model_checkpoints)
                
        # Sort by timestamp
        checkpoints = sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
        
        # Convert to dict and limit
        for checkpoint in checkpoints[:limit]:
            history.append(checkpoint.to_dict())
            
        return history
        
    async def compare_checkpoints(self,
                                model_name: str,
                                checkpoint_id1: str,
                                checkpoint_id2: str) -> Dict[str, Any]:
        """Compare two checkpoints"""
        try:
            # Load metadata for both checkpoints
            meta1 = await self._get_checkpoint_metadata(model_name, checkpoint_id1)
            meta2 = await self._get_checkpoint_metadata(model_name, checkpoint_id2)
            
            if not meta1 or not meta2:
                return {'error': 'One or both checkpoints not found'}
                
            # Compare performance metrics
            perf_comparison = {}
            for metric in set(meta1.performance_metrics.keys()) | set(meta2.performance_metrics.keys()):
                val1 = meta1.performance_metrics.get(metric, 0)
                val2 = meta2.performance_metrics.get(metric, 0)
                perf_comparison[metric] = {
                    'checkpoint1': val1,
                    'checkpoint2': val2,
                    'difference': val2 - val1,
                    'improvement': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
                
            # Compare overfitting metrics
            overfit_comparison = {}
            if meta1.overfitting_metrics and meta2.overfitting_metrics:
                for metric in set(meta1.overfitting_metrics.keys()) | set(meta2.overfitting_metrics.keys()):
                    val1 = meta1.overfitting_metrics.get(metric, 0)
                    val2 = meta2.overfitting_metrics.get(metric, 0)
                    overfit_comparison[metric] = {
                        'checkpoint1': val1,
                        'checkpoint2': val2,
                        'difference': val2 - val1
                    }
                    
            return {
                'checkpoint1': {
                    'id': checkpoint_id1,
                    'timestamp': meta1.timestamp,
                    'status': meta1.status.value
                },
                'checkpoint2': {
                    'id': checkpoint_id2,
                    'timestamp': meta2.timestamp,
                    'status': meta2.status.value
                },
                'performance_comparison': perf_comparison,
                'overfitting_comparison': overfit_comparison,
                'time_difference': meta2.timestamp - meta1.timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to compare checkpoints: {e}")
            return {'error': str(e)}
            
    async def export_checkpoint(self,
                              model_name: str,
                              checkpoint_id: str,
                              export_path: str) -> bool:
        """Export checkpoint to external location"""
        try:
            checkpoint_path = self.checkpoint_dir / model_name / checkpoint_id
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
                
            # Create export directory
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy checkpoint files
            shutil.copytree(checkpoint_path, export_dir / checkpoint_id, dirs_exist_ok=True)
            
            logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return False
            
    async def import_checkpoint(self,
                              model_name: str,
                              import_path: str) -> Optional[str]:
        """Import checkpoint from external location"""
        try:
            import_dir = Path(import_path)
            
            if not import_dir.exists():
                logger.error(f"Import path not found: {import_path}")
                return None
                
            # Load metadata
            metadata_path = import_dir / METADATA_FILENAME
            metadata = await self._load_checkpoint_metadata(import_dir)
            
            if not metadata:
                logger.error("Invalid checkpoint metadata")
                return None
                
            # Copy to checkpoint directory
            checkpoint_path = self.checkpoint_dir / model_name / metadata.checkpoint_id
            shutil.copytree(import_dir, checkpoint_path, dirs_exist_ok=True)
            
            # Update registry
            self.checkpoints[model_name].append(metadata)
            
            logger.info(f"Imported checkpoint {metadata.checkpoint_id}")
            return metadata.checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to import checkpoint: {e}")
            return None
            
    # Private helper methods
    def _generate_checkpoint_id(self, model_name: str) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{model_name}_{timestamp}_{random_suffix}"
        
    async def _save_state(self, state: Dict[str, Any], path: Path) -> None:
        """Save state to file"""
        # Handle different state types
        if 'torch_state' in state:
            # Save PyTorch models separately
            torch_path = path.with_suffix('.pth')
            torch.save(state['torch_state'], torch_path)
            state['torch_state'] = str(torch_path.name)
            
        # Save main state
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    async def _load_state(self, path: Path) -> Dict[str, Any]:
        """Load state from file"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        # Load PyTorch models if referenced
        if 'torch_state' in state and isinstance(state['torch_state'], str):
            torch_path = path.parent / state['torch_state']
            if torch_path.exists():
                state['torch_state'] = torch.load(torch_path)
                
        return state
        
    async def _save_metadata(self, metadata: CheckpointMetadata, path: Path) -> None:
        """Save metadata to file"""
        await self._save_json(metadata.to_dict(), path)
        
    async def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save JSON data to file"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    async def _load_checkpoint_metadata(self, checkpoint_path: Path) -> Optional[CheckpointMetadata]:
        """Load checkpoint metadata"""
        try:
            metadata_path = checkpoint_path / METADATA_FILENAME
            
            if not metadata_path.exists():
                return None
                
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                
            return CheckpointMetadata.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
            
    async def _get_checkpoint_metadata(self, 
                                     model_name: str, 
                                     checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get checkpoint metadata from registry"""
        for metadata in self.checkpoints.get(model_name, []):
            if metadata.checkpoint_id == checkpoint_id:
                return metadata
        return None
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for integrity check"""
        hash_md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
                
        return hash_md5.hexdigest()
        
    async def _verify_checkpoint_integrity(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint file integrity"""
        try:
            # Load metadata
            metadata = await self._load_checkpoint_metadata(checkpoint_path)
            
            if not metadata:
                return False
                
            # Verify file hashes
            for filename, expected_hash in metadata.file_hashes.items():
                file_path = checkpoint_path / filename
                
                if not file_path.exists():
                    logger.error(f"Missing checkpoint file: {filename}")
                    return False
                    
                actual_hash = self._calculate_file_hash(file_path)
                
                if actual_hash != expected_hash:
                    logger.error(f"Hash mismatch for {filename}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False
            
    def _validate_metadata(self, metadata: CheckpointMetadata) -> bool:
        """Validate metadata consistency"""
        # Check required fields
        required_fields = ['checkpoint_id', 'model_name', 'timestamp']
        
        for field in required_fields:
            if not getattr(metadata, field, None):
                return False
                
        # Check timestamp validity
        if metadata.timestamp > time.time() + 86400:  # Future timestamp
            return False
            
        return True
        
    def _evaluate_performance_metrics(self, metrics: Dict[str, float]) -> float:
        """Evaluate performance metrics quality"""
        score = 1.0
        
        # Check win rate
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.4:
            score *= 0.5
        elif win_rate > 0.8:
            score *= 0.8  # Too good to be true?
            
        # Check profit factor
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor < 1.0:
            score *= 0.3
        elif profit_factor > 3.0:
            score *= 0.9  # Suspicious
            
        # Check drawdown
        max_drawdown = metrics.get('max_drawdown', 0)
        if abs(max_drawdown) > 0.2:  # 20% drawdown
            score *= 0.7
            
        return max(0.0, min(1.0, score))
        
    def _evaluate_overfitting_metrics(self, metrics: Dict[str, float]) -> float:
        """Evaluate overfitting metrics"""
        score = 1.0
        
        # Check performance gap
        perf_gap = metrics.get('performance_gap', 0)
        if perf_gap > 0.2:
            score *= 0.3
        elif perf_gap > 0.1:
            score *= 0.7
            
        # Check confidence calibration
        calibration_error = metrics.get('confidence_calibration_error', 0)
        if calibration_error > 0.3:
            score *= 0.5
            
        # Check feature stability
        feature_stability = metrics.get('feature_stability_score', 1.0)
        score *= feature_stability
        
        return max(0.0, min(1.0, score))
        
    async def _cleanup_old_checkpoints(self, model_name: str) -> None:
        """Remove old checkpoints beyond limit"""
        checkpoints = self.checkpoints.get(model_name, [])
        
        if len(checkpoints) <= MAX_CHECKPOINTS_PER_MODEL:
            return
            
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.timestamp)
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:len(checkpoints) - MAX_CHECKPOINTS_PER_MODEL]
        
        for metadata in to_remove:
            try:
                # Skip if active
                if metadata.checkpoint_id == self.active_checkpoints.get(model_name):
                    continue
                    
                # Remove files
                checkpoint_path = self.checkpoint_dir / model_name / metadata.checkpoint_id
                
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    
                # Remove from registry
                self.checkpoints[model_name].remove(metadata)
                
                logger.info(f"Removed old checkpoint: {metadata.checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Failed to remove checkpoint: {e}")
                
    async def _load_checkpoint_registry(self) -> None:
        """Load existing checkpoints from disk"""
        try:
            # Scan checkpoint directory
            for model_dir in self.checkpoint_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    
                    for checkpoint_dir in model_dir.iterdir():
                        if checkpoint_dir.is_dir():
                            # Load metadata
                            metadata = await self._load_checkpoint_metadata(checkpoint_dir)
                            
                            if metadata:
                                self.checkpoints[model_name].append(metadata)
                                
                                # Set active if marked
                                if metadata.status == CheckpointStatus.ACTIVE:
                                    self.active_checkpoints[model_name] = metadata.checkpoint_id
                                    
            logger.info(f"Loaded {sum(len(v) for v in self.checkpoints.values())} checkpoints")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint registry: {e}")


# Example component implementation
class ExampleCheckpointableModel(BaseCheckpointable):
    """Example of a checkpointable model"""
    
    def __init__(self):
        self.model_state = {'weights': np.random.randn(10, 10)}
        self.training_epoch = 0
        self.best_score = 0.0
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'model_state': self.model_state,
            'training_epoch': self.training_epoch,
            'best_score': self.best_score,
            'timestamp': time.time()
        }
        
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state"""
        self.model_state = state['model_state']
        self.training_epoch = state['training_epoch']
        self.best_score = state['best_score']
        
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """Get metadata"""
        return {
            'training_epoch': self.training_epoch,
            'model_type': 'example',
            'parameters': {'size': self.model_state['weights'].shape}
        }
        
    def get_overfitting_metrics(self) -> Dict[str, float]:
        """Get overfitting metrics"""
        return {
            'performance_gap': 0.05,
            'confidence_calibration_error': 0.1,
            'feature_stability_score': 0.95
        }


# Example usage
async def example_usage():
    """Example of using CheckpointManager"""
    
    # Initialize manager
    manager = CheckpointManager("./checkpoints")
    
    # Create example model
    model = ExampleCheckpointableModel()
    
    # Save checkpoint
    checkpoint_id = await manager.save_checkpoint(
        model_name="example_model",
        component=model,
        performance_metrics={
            'win_rate': 0.65,
            'profit_factor': 1.5,
            'max_drawdown': -0.1
        }
    )
    
    print(f"Saved checkpoint: {checkpoint_id}")
    
    # Modify model
    model.training_epoch = 100
    model.best_score = 0.95
    
    # Save another checkpoint
    checkpoint_id2 = await manager.save_checkpoint(
        model_name="example_model",
        component=model,
        performance_metrics={
            'win_rate': 0.68,
            'profit_factor': 1.6,
            'max_drawdown': -0.08
        }
    )
    
    # Validate checkpoint
    validation = await manager.validate_checkpoint("example_model", checkpoint_id2)
    print(f"Validation result: {validation.is_valid}, score: {validation.validation_score}")
    
    # Compare checkpoints
    comparison = await manager.compare_checkpoints(
        "example_model",
        checkpoint_id,
        checkpoint_id2
    )
    print(f"Checkpoint comparison: {comparison}")
    
    # Rollback to first checkpoint
    success = await manager.rollback_to_checkpoint(
        "example_model",
        model,
        checkpoint_id,
        reason="Testing rollback"
    )
    
    print(f"Rollback success: {success}")
    print(f"Model epoch after rollback: {model.training_epoch}")
    
    # Get history
    history = await manager.get_checkpoint_history("example_model")
    print(f"Checkpoint history: {len(history)} checkpoints")


if __name__ == "__main__":
    asyncio.run(example_usage())