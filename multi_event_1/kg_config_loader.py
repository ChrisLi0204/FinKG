"""
Knowledge Graph Configuration Loader (Simplified for v3)
=========================================================

Loads KG configuration from YAML file and provides easy access to:
- Asset keywords and metadata
- Event keywords and metadata  
- Movement indicators
- Relation definitions

Usage:
    from kg_config_loader import load_kg_config
    
    config = load_kg_config()  # or load_kg_config('path/to/kg_config.yaml')
    
    # Access data
    config.ASSET_KEYWORDS      # {asset_id: [keywords]}
    config.EVENT_KEYWORDS      # {event_id: [keywords]}
    config.MOVEMENT_INDICATORS # {strength: [words]}
    config.ASSET_TYPE_MAP      # {asset_id: type}
    config.ASSET_DISPLAY_NAMES # {asset_id: display_name}
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional


class KGConfig:
    """
    Simplified configuration class for Knowledge Graph extraction.
    Loads from YAML and provides dict-style access for v3 compatibility.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to kg_config.yaml. If None, looks in same directory.
        """
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "kg_config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
        self._build_dicts()
    
    def _load_config(self):
        """Load the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
        
        self.version = self._raw_config.get('version', '1.0')
        self.last_updated = self._raw_config.get('last_updated', 'unknown')
    
    def _build_dicts(self):
        """Build dictionaries for easy access."""
        
        # =====================================================================
        # ASSET DATA
        # =====================================================================
        self.ASSET_KEYWORDS: Dict[str, List[str]] = {}
        self.ASSET_TYPE_MAP: Dict[str, str] = {}
        self.ASSET_DISPLAY_NAMES: Dict[str, str] = {}
        
        for asset_id, asset_data in self._raw_config.get('assets', {}).items():
            self.ASSET_KEYWORDS[asset_id] = asset_data.get('keywords', [])
            self.ASSET_TYPE_MAP[asset_id] = asset_data.get('type', 'unknown')
            self.ASSET_DISPLAY_NAMES[asset_id] = asset_data.get('display_name', asset_id)
        
        # =====================================================================
        # EVENT DATA
        # =====================================================================
        self.EVENT_KEYWORDS: Dict[str, List[str]] = {}
        self.EVENT_DISPLAY_NAMES: Dict[str, str] = {}
        self.EVENT_QUALIFIERS: Dict[str, Dict[str, List[str]]] = {}
        
        for event_id, event_data in self._raw_config.get('events', {}).items():
            self.EVENT_KEYWORDS[event_id] = event_data.get('keywords', [])
            self.EVENT_DISPLAY_NAMES[event_id] = event_data.get('display_name', event_id)
            self.EVENT_QUALIFIERS[event_id] = event_data.get('qualifiers', {})
        
        # =====================================================================
        # MOVEMENT INDICATORS
        # =====================================================================
        self.MOVEMENT_INDICATORS: Dict[str, List[str]] = self._raw_config.get('movement_indicators', {})
        
        # Flattened positive/negative/neutral lists for convenience
        self.POSITIVE_KEYWORDS: List[str] = (
            self.MOVEMENT_INDICATORS.get('strong_positive', []) +
            self.MOVEMENT_INDICATORS.get('positive', [])
        )
        self.NEGATIVE_KEYWORDS: List[str] = (
            self.MOVEMENT_INDICATORS.get('strong_negative', []) +
            self.MOVEMENT_INDICATORS.get('negative', [])
        )
        self.NEUTRAL_KEYWORDS: List[str] = self.MOVEMENT_INDICATORS.get('neutral', [])
        
        # =====================================================================
        # RELATION DATA
        # =====================================================================
        self.RELATION_KEYWORDS: Dict[str, List[str]] = {}
        
        for rel_name, rel_data in self._raw_config.get('relations', {}).items():
            indicators = rel_data.get('indicators', {})
            if isinstance(indicators, dict):
                # Flatten nested indicators (strong/moderate/weak)
                all_indicators = []
                for strength_indicators in indicators.values():
                    if isinstance(strength_indicators, list):
                        all_indicators.extend(strength_indicators)
                self.RELATION_KEYWORDS[rel_name] = all_indicators
            elif isinstance(indicators, list):
                self.RELATION_KEYWORDS[rel_name] = indicators
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def get_asset_type(self, asset_id: str) -> str:
        """Get the type of an asset."""
        return self.ASSET_TYPE_MAP.get(asset_id, 'unknown')
    
    def get_asset_display_name(self, asset_id: str) -> str:
        """Get the display name of an asset."""
        return self.ASSET_DISPLAY_NAMES.get(asset_id, asset_id.replace('_', ' ').title())
    
    def get_event_display_name(self, event_id: str) -> str:
        """Get the display name of an event."""
        return self.EVENT_DISPLAY_NAMES.get(event_id, event_id.replace('_', ' ').title())
    
    def get_event_qualifiers(self, event_id: str) -> Dict[str, List[str]]:
        """Get qualifiers (positive/negative/neutral indicators) for an event."""
        return self.EVENT_QUALIFIERS.get(event_id, {})
    
    def __repr__(self) -> str:
        return (
            f"KGConfig(version={self.version}, "
            f"assets={len(self.ASSET_KEYWORDS)}, "
            f"events={len(self.EVENT_KEYWORDS)}, "
            f"relations={len(self.RELATION_KEYWORDS)})"
        )


# =============================================================================
# MODULE-LEVEL LOADER FUNCTION
# =============================================================================

_config_instance: Optional[KGConfig] = None


def load_kg_config(config_path: Optional[str] = None, reload: bool = False) -> KGConfig:
    """
    Load KG configuration (singleton pattern for efficiency).
    
    Args:
        config_path: Path to YAML config file. None uses default location.
        reload: If True, force reload even if already loaded.
    
    Returns:
        KGConfig instance
    """
    global _config_instance
    
    if _config_instance is None or reload:
        _config_instance = KGConfig(config_path)
    
    return _config_instance


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    config = load_kg_config()
    print(f"Loaded: {config}")
    
    print(f"\n📦 Assets ({len(config.ASSET_KEYWORDS)}):")
    for asset_id, keywords in list(config.ASSET_KEYWORDS.items())[:5]:
        print(f"  {asset_id}: {keywords[:3]}...")
    
    print(f"\n📅 Events ({len(config.EVENT_KEYWORDS)}):")
    for event_id, keywords in config.EVENT_KEYWORDS.items():
        print(f"  {event_id}: {keywords[:3]}...")
    
    print(f"\n📈 Movement Indicators:")
    for strength, words in config.MOVEMENT_INDICATORS.items():
        print(f"  {strength}: {words[:5]}...")
    
    print(f"\n✅ Config loaded successfully!")
