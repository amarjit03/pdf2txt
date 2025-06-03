"""
Memory Manager - Real-time Memory Monitoring and Control
Actively manages memory usage during PDF processing to prevent crashes
"""

import gc
import psutil
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging
import weakref
import tempfile
import shutil
from pathlib import Path

from pdf2text.config import get_config


class MemoryState(Enum):
    """System memory states"""
    HEALTHY = "healthy"        # < 60% memory used
    CAUTIOUS = "cautious"      # 60-75% memory used  
    WARNING = "warning"        # 75-85% memory used
    CRITICAL = "critical"      # 85-95% memory used
    EMERGENCY = "emergency"    # > 95% memory used


class CleanupPriority(Enum):
    """Cleanup operation priorities"""
    LOW = 1         # Optional cleanup
    MEDIUM = 2      # Recommended cleanup
    HIGH = 3        # Required cleanup
    EMERGENCY = 4   # Immediate cleanup needed


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot"""
    timestamp: float
    total_mb: float
    available_mb: float
    used_mb: float
    percentage: float
    state: MemoryState
    
    # Process-specific memory
    process_mb: float
    process_percentage: float
    
    # Memory pressure indicators
    swap_used_mb: float = 0.0
    memory_pressure_score: float = 0.0


@dataclass 
class ManagedResource:
    """Trackable resource for cleanup"""
    resource_id: str
    resource_type: str          # "pdf_document", "image_buffer", "text_cache"
    size_mb: float
    created_at: float
    last_accessed: float
    cleanup_callback: Optional[Callable] = None
    priority: CleanupPriority = CleanupPriority.MEDIUM
    metadata: Dict = field(default_factory=dict)


class MemoryManager:
    """Intelligent memory management and resource control"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Memory thresholds (percentages)
        self.thresholds = {
            MemoryState.HEALTHY: 60,
            MemoryState.CAUTIOUS: 75,
            MemoryState.WARNING: 85,
            MemoryState.CRITICAL: 95
        }
        
        # Resource tracking
        self.managed_resources: Dict[str, ManagedResource] = {}
        self.temp_directories: List[Path] = []
        self.cleanup_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.memory_history: List[MemorySnapshot] = []
        self.max_history_size = 100
        
        # Safety limits
        self.max_memory_mb = self.config.memory.max_memory_mb
        self.emergency_callback: Optional[Callable] = None
        
        # Performance tracking
        self.gc_stats = {
            'manual_collections': 0,
            'emergency_collections': 0,
            'resources_cleaned': 0,
            'memory_freed_mb': 0.0
        }
        
        self.logger.info(f"Memory Manager initialized with {self.max_memory_mb}MB limit")
    
    def start_monitoring(self, interval_seconds: float = 2.0):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Continuous monitoring loop (runs in background thread)"""
        while self.monitoring_active:
            try:
                snapshot = self.get_memory_snapshot()
                self._handle_memory_state(snapshot)
                
                # Keep memory history for analysis
                self.memory_history.append(snapshot)
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval_seconds * 2)  # Back off on errors
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory state snapshot"""
        # System memory
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        # Process memory
        process = psutil.Process()
        process_info = process.memory_info()
        
        # Calculate memory state
        percentage = memory_info.percent
        if percentage < self.thresholds[MemoryState.HEALTHY]:
            state = MemoryState.HEALTHY
        elif percentage < self.thresholds[MemoryState.CAUTIOUS]:
            state = MemoryState.CAUTIOUS
        elif percentage < self.thresholds[MemoryState.WARNING]:
            state = MemoryState.WARNING
        elif percentage < self.thresholds[MemoryState.CRITICAL]:
            state = MemoryState.CRITICAL
        else:
            state = MemoryState.EMERGENCY
        
        # Memory pressure score (0.0 = no pressure, 1.0 = maximum pressure)
        pressure_score = min(1.0, (percentage - 50) / 50) if percentage > 50 else 0.0
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_mb=memory_info.total / (1024 * 1024),
            available_mb=memory_info.available / (1024 * 1024),
            used_mb=memory_info.used / (1024 * 1024),
            percentage=percentage,
            state=state,
            process_mb=process_info.rss / (1024 * 1024),
            process_percentage=(process_info.rss / memory_info.total) * 100,
            swap_used_mb=swap_info.used / (1024 * 1024),
            memory_pressure_score=pressure_score
        )
    
    def _handle_memory_state(self, snapshot: MemorySnapshot):
        """Handle different memory states with appropriate actions"""
        if snapshot.state == MemoryState.HEALTHY:
            # All good, no action needed
            pass
            
        elif snapshot.state == MemoryState.CAUTIOUS:
            # Light cleanup of low-priority resources
            self._trigger_cleanup(CleanupPriority.LOW)
            
        elif snapshot.state == MemoryState.WARNING:
            # More aggressive cleanup
            self._trigger_cleanup(CleanupPriority.MEDIUM)
            self.logger.warning(f"Memory usage high: {snapshot.percentage:.1f}%")
            
        elif snapshot.state == MemoryState.CRITICAL:
            # Emergency cleanup
            self._trigger_cleanup(CleanupPriority.HIGH)
            self.force_garbage_collection()
            self.logger.error(f"Memory usage critical: {snapshot.percentage:.1f}%")
            
        elif snapshot.state == MemoryState.EMERGENCY:
            # Last resort actions
            self._emergency_cleanup()
            self.logger.critical(f"Memory emergency: {snapshot.percentage:.1f}%")
            
            # Call emergency callback if set
            if self.emergency_callback:
                try:
                    self.emergency_callback(snapshot)
                except Exception as e:
                    self.logger.error(f"Emergency callback failed: {e}")
    
    def register_resource(self, resource: ManagedResource) -> str:
        """Register a resource for memory management"""
        self.managed_resources[resource.resource_id] = resource
        self.logger.debug(f"Registered resource: ID='{resource.resource_id}', Type='{resource.resource_type}', SizeMB={resource.size_mb:.1f}, Metadata={resource.metadata}. Total managed resources: {len(self.managed_resources)}")
        return resource.resource_id
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource"""
        if resource_id in self.managed_resources:
            resource = self.managed_resources.pop(resource_id)
            self.logger.debug(f"Unregistered resource: ID='{resource.resource_id}', Type='{resource.resource_type}', SizeMB={resource.size_mb:.1f}. Total managed resources: {len(self.managed_resources)}")
            return True
        return False
    
    def update_resource_access(self, resource_id: str):
        """Update last access time for a resource"""
        if resource_id in self.managed_resources:
            self.managed_resources[resource_id].last_accessed = time.time()
    
    def _trigger_cleanup(self, min_priority: CleanupPriority):
        """Trigger cleanup of resources at or above priority level"""
        cleaned_count = 0
        freed_mb = 0.0
        
        # Sort resources by priority and age (oldest first)
        candidates = [
            (resource_id, resource) 
            for resource_id, resource in self.managed_resources.items()
            if resource.priority.value >= min_priority.value
        ]
        
        # Sort by priority (high first) then by age (old first)
        candidates.sort(key=lambda x: (x[1].priority.value, x[1].created_at))
        
        for resource_id, resource in candidates:
            self.logger.debug(f"Attempting cleanup for resource: ID='{resource_id}', Type='{resource.resource_type}', Priority='{resource.priority.name}'")
            try:
                if resource.cleanup_callback:
                    resource.cleanup_callback()
                    freed_mb += resource.size_mb
                    cleaned_count += 1
                    self.unregister_resource(resource_id)
                    self.logger.info(f"Successfully cleaned up resource: ID='{resource_id}', Type='{resource.resource_type}', FreedMB={resource.size_mb:.1f}")
                else:
                    self.logger.debug(f"Resource ID='{resource_id}' has no cleanup_callback or was not cleaned.")
            except Exception as e:
                self.logger.error(f"Cleanup failed for {resource_id}: {e}")
        
        if cleaned_count > 0:
            self.gc_stats['resources_cleaned'] += cleaned_count
            self.gc_stats['memory_freed_mb'] += freed_mb
            self.logger.info(f"Cleaned {cleaned_count} resources, freed {freed_mb:.1f}MB")
    
    def _emergency_cleanup(self):
        """Emergency cleanup - clean everything possible"""
        self.logger.critical("Performing emergency memory cleanup")
        
        # Clean all managed resources
        self._trigger_cleanup(CleanupPriority.LOW)  # Clean everything
        
        # Force garbage collection multiple times
        for i in range(3):
            collected = gc.collect()
            self.logger.debug(f"Emergency GC #{i+1}: collected {collected} objects")
        
        # Clean temporary directories
        self._cleanup_temp_directories()
        
        # Update stats
        self.gc_stats['emergency_collections'] += 1
    
    def force_garbage_collection(self) -> int:
        """Force Python garbage collection"""
        collected = gc.collect()
        self.gc_stats['manual_collections'] += 1
        self.logger.debug(f"Manual GC: collected {collected} objects")
        return collected
    
    def create_temp_directory(self, prefix: str = "pdf_agent_") -> Path:
        """Create managed temporary directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_directories.append(temp_dir)
        self.logger.debug(f"Created temp directory: {temp_dir}")
        return temp_dir
    
    def _cleanup_temp_directories(self):
        """Clean up all managed temporary directories"""
        cleaned = 0
        for temp_dir in self.temp_directories[:]:  # Copy list to avoid modification during iteration
            self.logger.debug(f"Attempting to remove temp directory: {temp_dir}")
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    cleaned += 1
                self.temp_directories.remove(temp_dir)
            except Exception as e:
                self.logger.error(f"Failed to cleanup temp dir {temp_dir}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} temporary directories")
    
    def check_can_allocate(self, required_mb: float) -> bool:
        """Check if we can safely allocate the requested memory"""
        snapshot = self.get_memory_snapshot()
        
        # Check against our configured limit
        if snapshot.process_mb + required_mb > self.max_memory_mb:
            return False
        
        # Check against system availability
        if required_mb > snapshot.available_mb * 0.8:  # Leave 20% buffer
            return False
        
        # Check if this would push us into critical state
        projected_usage = ((snapshot.used_mb + required_mb) / snapshot.total_mb) * 100
        if projected_usage > self.thresholds[MemoryState.CRITICAL]:
            return False
        
        return True
    
    def wait_for_memory(self, required_mb: float, timeout_seconds: float = 30) -> bool:
        """Wait for sufficient memory to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            if self.check_can_allocate(required_mb):
                return True
            
            # Try cleanup to free memory
            self._trigger_cleanup(CleanupPriority.MEDIUM)
            self.force_garbage_collection()
            
            time.sleep(1.0)
        
        return False
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        snapshot = self.get_memory_snapshot()
        
        total_managed_mb = sum(r.size_mb for r in self.managed_resources.values())
        
        return {
            'current_state': snapshot.state.value,
            'system_total_mb': snapshot.total_mb,
            'system_used_mb': snapshot.used_mb,
            'system_available_mb': snapshot.available_mb,
            'system_percentage': snapshot.percentage,
            'process_mb': snapshot.process_mb,
            'managed_resources_count': len(self.managed_resources),
            'managed_resources_mb': total_managed_mb,
            'temp_directories': len(self.temp_directories),
            'memory_pressure': snapshot.memory_pressure_score,
            'gc_stats': self.gc_stats.copy()
        }
    
    def get_memory_trend(self, minutes: int = 5) -> Dict:
        """Analyze memory usage trend over time"""
        if not self.memory_history:
            return {'trend': 'unknown', 'data_points': 0}
        
        # Filter recent history
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.memory_history if s.timestamp > cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {'trend': 'insufficient_data', 'data_points': len(recent_snapshots)}
        
        # Calculate trend
        first_usage = recent_snapshots[0].percentage
        last_usage = recent_snapshots[-1].percentage
        change = last_usage - first_usage
        
        if change < -2:
            trend = 'decreasing'
        elif change > 2:
            trend = 'increasing'
        else:
            trend = 'stable'
        
        # Calculate average pressure
        avg_pressure = sum(s.memory_pressure_score for s in recent_snapshots) / len(recent_snapshots)
        
        return {
            'trend': trend,
            'change_percentage': change,
            'data_points': len(recent_snapshots),
            'avg_pressure': avg_pressure,
            'current_vs_start': last_usage - first_usage
        }
    
    def set_emergency_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Set callback to be called during memory emergencies"""
        self.emergency_callback = callback
    
    def cleanup_and_shutdown(self):
        """Clean shutdown - cleanup all resources"""
        self.logger.info("Memory Manager shutting down")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clean all resources
        self._emergency_cleanup()
        
        # Final garbage collection
        self.force_garbage_collection()
        
        self.logger.info("Memory Manager shutdown complete")


# Context manager for memory-managed operations
class MemoryContext:
    """Context manager for automatic memory management during operations"""
    
    def __init__(self, memory_manager: MemoryManager, 
                operation_name: str, 
                estimated_mb: float = 0):
        self.memory_manager = memory_manager
        self.operation_name = operation_name  
        self.estimated_mb = estimated_mb
        self.start_snapshot: Optional[MemorySnapshot] = None
        self.resources_created: List[str] = []
    
    def __enter__(self):
        self.start_snapshot = self.memory_manager.get_memory_snapshot()
        self.memory_manager.logger.debug(f"MemoryContext '{self.operation_name}' entering: ProcessMB={self.start_snapshot.process_mb:.1f}, AvailableMB={self.start_snapshot.available_mb:.1f}, SysUsagePercent={self.start_snapshot.percentage:.1f}%")
        
        # Check if we have enough memory
        if self.estimated_mb > 0 and not self.memory_manager.check_can_allocate(self.estimated_mb):
            # Try to free memory
            if not self.memory_manager.wait_for_memory(self.estimated_mb, timeout_seconds=10):
                self.memory_manager.logger.warning(f"MemoryContext '{self.operation_name}': Insufficient memory after waiting. RequiredMB={self.estimated_mb:.1f}, StartProcessMB={self.start_snapshot.process_mb:.1f}, AvailableMB={self.memory_manager.get_memory_snapshot().available_mb:.1f}")
                raise MemoryError(f"Insufficient memory for {self.operation_name}: need {self.estimated_mb:.1f}MB")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup any resources created during this operation
        for resource_id in self.resources_created:
            self.memory_manager.unregister_resource(resource_id)
        
        # Force cleanup if memory pressure increased significantly
        end_snapshot = self.memory_manager.get_memory_snapshot()
        memory_increase_mb = end_snapshot.process_mb - self.start_snapshot.process_mb
        self.memory_manager.logger.debug(f"MemoryContext '{self.operation_name}' exiting: StartProcessMB={self.start_snapshot.process_mb:.1f}, EndProcessMB={end_snapshot.process_mb:.1f}, DiffMB={memory_increase_mb:.1f}. SysUsagePercent={end_snapshot.percentage:.1f}%, AvailableMB={end_snapshot.available_mb:.1f}")

        memory_increase = end_snapshot.percentage - self.start_snapshot.percentage # This is system percentage increase
        
        if memory_increase > 10:  # More than 10% increase
            self.memory_manager.logger.info(f"MemoryContext '{self.operation_name}': Significant memory increase (System usage {memory_increase:.1f} percentage points, ProcessMB from {self.start_snapshot.process_mb:.1f} to {end_snapshot.process_mb:.1f}). Triggering medium cleanup.")
            self.memory_manager._trigger_cleanup(CleanupPriority.MEDIUM)
            self.memory_manager.force_garbage_collection()
    
    def register_resource(self, resource: ManagedResource) -> str:
        """Register a resource within this context"""
        resource_id = self.memory_manager.register_resource(resource)
        self.resources_created.append(resource_id)
        return resource_id


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def memory_managed_operation(estimated_mb: float = 0):
    """Decorator for memory-managed operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            memory_manager = get_memory_manager()
            with MemoryContext(memory_manager, func.__name__, estimated_mb):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test memory manager
    import time
    
    manager = MemoryManager()
    
    print("=== Memory Manager Test ===")
    print("Starting monitoring...")
    manager.start_monitoring(interval_seconds=1.0)
    
    # Test resource registration
    test_resource = ManagedResource(
        resource_id="test_1",
        resource_type="test_buffer",
        size_mb=10.0,
        created_at=time.time(),
        last_accessed=time.time(),
        cleanup_callback=lambda: print("Cleaning up test resource"),
        priority=CleanupPriority.LOW
    )
    
    manager.register_resource(test_resource)
    
    # Show stats
    stats = manager.get_memory_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test context manager
    print("\nTesting memory context...")
    with MemoryContext(manager, "test_operation", 50.0) as ctx:
        print("Inside memory-managed context")
        time.sleep(2)
    
    print("Shutting down...")
    manager.cleanup_and_shutdown()