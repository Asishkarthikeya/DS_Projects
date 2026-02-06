"""
Configuration manager for CLI settings
"""
import json
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """Manages CLI configuration"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".code-crawler"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create default
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> dict:
        """Get default configuration"""
        return {
            "model": "gemini-2.5-flash",
            "provider": "gemini",
            "vector_db": "chroma",
            "max_context_tokens": 100000,
            "temperature": 0.7,
            "use_agent": True,
            "use_reranking": True,
        }
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        self.config[key] = value
        self._save_config()
    
    def delete(self, key: str):
        """Delete a configuration value"""
        if key in self.config:
            del self.config[key]
            self._save_config()
    
    def list_all(self) -> dict:
        """Get all configuration values"""
        return self.config.copy()
    
    def reset(self):
        """Reset to default configuration"""
        self.config = self._default_config()
        self._save_config()
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider"""
        # First check environment variables
        env_keys = {
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        
        env_key = env_keys.get(provider.lower())
        if env_key:
            api_key = os.getenv(env_key)
            if api_key:
                return api_key
        
        # Then check config file
        return self.config.get(f"{provider.upper()}_API_KEY")
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider"""
        self.config[f"{provider.upper()}_API_KEY"] = api_key
        self._save_config()
