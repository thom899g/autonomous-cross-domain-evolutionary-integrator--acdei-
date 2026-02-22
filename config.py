"""
ACDEI Configuration Management
Centralized configuration with environment-based settings and validation
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('acdei.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Firebase Firestore configuration"""
    project_id: str = "acdei-system"
    credentials_path: Optional[str] = None
    collection_prefix: str = "acdei_"
    timeout_seconds: int = 30
    max_retries: int = 3

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    batch_size: int = 1000
    max_workers: int = 4
    chunk_size: int = 100
    default_sample_size: int = 10000
    similarity_threshold: float = 0.7
    correlation_threshold: float = 0.3

@dataclass
class GraphConfig:
    """Knowledge graph configuration"""
    min_node_degree: int = 2
    max_cluster_size: int = 50
    prune_threshold: float = 0.1
    community_resolution: float = 1.0

class ConfigManager:
    """Central configuration manager with validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.ConfigManager")
        self.config_path = config_path or "acdei_config.json"
        
        # Initialize with defaults
        self.db_config = DatabaseConfig()
        self.proc_config = ProcessingConfig()
        self.graph_config = GraphConfig()
        
        # Load from file if exists
        self._load_config()
        
        # Validate configurations
        self._validate_configs()
        
        self.logger.info("Configuration initialized successfully")
    
    def _load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                # Update configurations from file
                if 'database' in data:
                    self.db_config = DatabaseConfig(**data['database'])
                if 'processing' in data:
                    self.proc_config = ProcessingConfig(**data['processing'])
                if 'graph' in data:
                    self.graph_config = GraphConfig(**data['graph'])
                    
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self.logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Error loading config: {e}. Using defaults.")
        except Exception as e:
            self.logger.error(f"Unexpected error loading config: {e}")
            raise
    
    def _validate_configs(self) -> None:
        """Validate all configuration values"""
        # Database validation
        assert self.db_config.timeout_seconds > 0, "Timeout must be positive"
        assert self.db_config.max_retries >= 0, "Max retries must be non-negative"
        
        # Processing validation
        assert 0 < self.proc_config.similarity_threshold <= 1, "Similarity threshold must be in (0,1]"
        assert -1 <= self.proc_config.correlation_threshold <= 1, "Correlation threshold must be in [-1,1]"
        assert self.proc_config.batch_size > 0, "Batch size must be positive"
        
        # Graph validation
        assert self.graph_config.min_node_degree >= 0, "Min node degree must be non-negative"
        assert self.graph_config.max_cluster_size > 0, "Max cluster size must be positive"
        
        self.logger.debug("Configuration validation passed")
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            config_data = {
                'database': self.db_config.__dict__,
                'processing': self.proc_config.__dict__,
                'graph': self.graph_config.__dict__
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            raise
    
    def get_firebase_credentials(self) -> Optional[Dict[str, Any]]:
        """Get Firebase credentials from environment or file"""
        # Check for environment variable
        creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON')
        if creds_json:
            try:
                return json.loads(creds_json)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in FIREBASE_CREDENTIALS_JSON: {e}")
        
        # Check for credentials file
        if self.db_config.credentials_path:
            creds_file = Path(self.db_config.credentials_path)
            if creds_file.exists():
                try:
                    with open(creds_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.error(f"Error reading credentials file: {e}")
        
        self.logger.warning("No Firebase credentials found")
        return None

# Global configuration instance
config = ConfigManager()