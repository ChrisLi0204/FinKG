"""
Multi-Event Causal Knowledge Graph Extraction - RELAXED VERSION
================================================================

This is a RELAXED version of multi_event_kg_v2.py that:
1. REMOVES the strict causal pattern requirement (Gate 3)
2. Extracts edges for ANY headline with BOTH event + asset keywords
3. Uses asset-aware direction extraction to handle multi-asset headlines
4. Integrates edges with same source, target, relation into single edge with multiple evidence

Key differences from multi_event_kg_v2.py:
- No CAUSAL_PATTERNS matching required
- Headlines with event+asset are ALWAYS extracted
- Edges with same (source, target, relation) are merged, with each news as evidence
- Supports event-to-event, event-to-asset, and asset-to-asset relationships
- No confidence scoring (removed)

This enables extracting headlines like:
  ✓ "Stocks soar on ecb bond-buying plans, jobs report optimism"
  ✓ "Dollar drifts lower with focus turning to us jobs report"
  ✓ "Fed rate cut, calmer politics to lift thai shares"
  ✓ "Labor market tightens as unemployment hits 4-year low"
"""

import re
import json
from collections import defaultdict
from datetime import datetime
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional

# Import keywords from YAML configuration
from kg_config_loader import load_kg_config

# Load configuration from YAML file
_config = load_kg_config()

# Extract keywords from config for module-level access
ASSET_KEYWORDS = _config.ASSET_KEYWORDS
ASSET_TYPE_MAP = _config.ASSET_TYPE_MAP
ASSET_DISPLAY_NAMES = _config.ASSET_DISPLAY_NAMES
EVENT_KEYWORDS = _config.EVENT_KEYWORDS
RELATION_KEYWORDS = _config.RELATION_KEYWORDS
POSITIVE_KEYWORDS = _config.POSITIVE_KEYWORDS
NEGATIVE_KEYWORDS = _config.NEGATIVE_KEYWORDS
NEUTRAL_KEYWORDS = _config.NEUTRAL_KEYWORDS

# Helper functions from config
get_asset_type = _config.get_asset_type
get_asset_display_name = _config.get_asset_display_name

# Event keyword lists for this module
RATE_CUT_KEYWORDS = EVENT_KEYWORDS.get('rate_cut', [])
RATE_HIKE_KEYWORDS = EVENT_KEYWORDS.get('rate_hike', [])
EMPLOYMENT_KEYWORDS = EVENT_KEYWORDS.get('employment', [])


# ============================================================================
# MECHANISM KEYWORDS (optional, for richer context)
# ============================================================================

MECHANISM_KEYWORDS = {
    'ahead_of_jobs': {
        'id': 'mech:ahead_of_jobs_report',
        'name': 'Ahead of Jobs Report',
        'type': 'Expectation_Timing',
        'patterns': [
            r'ahead of.*?(jobs report|employment data|nonfarm payrolls)',
            r'before.*?(jobs report|employment data)',
            r'(awaits?|awaiting|eyes on).*?(jobs report|employment data)',
        ]
    },
    'rate_cut_bets': {
        'id': 'mech:rate_cut_bets',
        'name': 'Rate Cut Expectations',
        'type': 'Policy_Expectation',
        'patterns': [
            r'rate cut (hopes?|bets?|speculation|optimism)',
            r'(hopes?|bets?).*?rate cut',
        ]
    },
}


# ============================================================================
# MOVEMENT INDICATORS (loaded from YAML config)
# ============================================================================

MOVEMENT_INDICATORS = _config.MOVEMENT_INDICATORS


# ============================================================================
# CORE DETECTION FUNCTIONS (RELAXED - No Causality Required)
# ============================================================================

def detect_event_type(text: str) -> Dict[str, bool]:
    """Detect which event types are mentioned in the text."""
    text_lower = text.lower()
    
    return {
        'rate_cut': any(keyword in text_lower for keyword in RATE_CUT_KEYWORDS),
        'rate_hike': any(keyword in text_lower for keyword in RATE_HIKE_KEYWORDS),
        'employment': any(keyword in text_lower for keyword in EMPLOYMENT_KEYWORDS)
    }


def detect_assets(text: str) -> Set[str]:
    """Detect all asset types mentioned in the text."""
    text_lower = text.lower()
    detected = set()
    
    for asset_id, keywords in ASSET_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r"'?s?\b"
            if re.search(pattern, text_lower):
                detected.add(asset_id)
                break
    
    return detected


def detect_mechanisms(text: str) -> Set[str]:
    """Detect mechanism/context nodes mentioned in the text."""
    detected = set()
    text_lower = text.lower()
    
    for mech_key, mech_config in MECHANISM_KEYWORDS.items():
        for pattern in mech_config['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.add(mech_config['id'])
                break
    
    return detected


def detect_employment_strength(text: str) -> Optional[str]:
    """Detect whether employment data is characterized as strong/weak/mixed."""
    text_lower = text.lower()
    
    strong_indicators = [
        r'strong.*job', r'robust.*job', r'blowout.*job', r'upbeat.*job',
        r'beat.*expect', r'exceed.*expect', r'better.*than.*expect',
        r'surprise', r'tight.*labor', r'labor.*tight', r'labor.*strengthen',
        r'claims.*fall', r'claims.*drop', r'payrolls.*beat', r'jobs.*beat'
    ]
    
    weak_indicators = [
        r'weak.*job', r'disappointing.*job', r'miss.*expect',
        r'soft.*job', r'worse.*than.*expect', r'below.*expect',
        r'claims.*rise', r'claims.*jump', r'unemployment.*rise',
        r'labor.*soften', r'labor.*weaken', r'job.*loss'
    ]
    
    has_strong = any(re.search(p, text_lower) for p in strong_indicators)
    has_weak = any(re.search(p, text_lower) for p in weak_indicators)
    
    if has_strong and has_weak:
        return 'mixed'
    elif has_strong:
        return 'strong'
    elif has_weak:
        return 'weak'
    
    return None


def get_all_movement_positions(text: str) -> List[Tuple[int, int, str, str]]:
    """Find all movement indicators in text with their positions."""
    text_lower = text.lower()
    movements = []
    
    direction_indicators = [
        ('strong_positive', MOVEMENT_INDICATORS['strong_positive']),
        ('positive', MOVEMENT_INDICATORS['positive']),
        ('strong_negative', MOVEMENT_INDICATORS['strong_negative']),
        ('negative', MOVEMENT_INDICATORS['negative']),
        ('neutral', MOVEMENT_INDICATORS['neutral']),
    ]
    
    for direction, indicators in direction_indicators:
        sorted_indicators = sorted(indicators, key=len, reverse=True)
        for indicator in sorted_indicators:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            for match in re.finditer(pattern, text_lower):
                if 'positive' in direction:
                    simple_dir = 'positive'
                elif 'negative' in direction:
                    simple_dir = 'negative'
                else:
                    simple_dir = 'neutral'
                movements.append((match.start(), match.end(), indicator, simple_dir))
    
    # Remove overlapping matches
    movements = sorted(movements, key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlapping = []
    last_end = -1
    for start, end, indicator, direction in movements:
        if start >= last_end:
            non_overlapping.append((start, end, indicator, direction))
            last_end = end
    
    return non_overlapping


def get_asset_positions(text: str) -> Dict[str, List[Tuple[int, int]]]:
    """Find all asset mentions in text with their positions."""
    text_lower = text.lower()
    asset_positions = {}
    
    for asset_id, keywords in ASSET_KEYWORDS.items():
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        for keyword in sorted_keywords:
            pattern = r'\b' + re.escape(keyword) + r"'?s?\b"
            for match in re.finditer(pattern, text_lower):
                if asset_id not in asset_positions:
                    asset_positions[asset_id] = []
                pos = (match.start(), match.end())
                if pos not in asset_positions[asset_id]:
                    asset_positions[asset_id].append(pos)
    
    return asset_positions


def infer_direction_from_clause(clause: str) -> str:
    """Infer direction from a single clause."""
    clause_lower = clause.lower()
    
    for indicator in MOVEMENT_INDICATORS['strong_negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'negative'
    
    for indicator in MOVEMENT_INDICATORS['negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'negative'
    
    for indicator in MOVEMENT_INDICATORS['strong_positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'positive'
    
    for indicator in MOVEMENT_INDICATORS['positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'positive'
    
    for indicator in MOVEMENT_INDICATORS['neutral']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'neutral'
    
    return 'neutral'


def extract_asset_movement_by_proximity(text: str) -> Dict[str, str]:
    """Match movements to assets using proximity/word distance."""
    text_lower = text.lower()
    asset_movements = {}
    
    asset_positions = get_asset_positions(text_lower)
    movement_positions = get_all_movement_positions(text_lower)
    
    if not movement_positions:
        return asset_movements
    
    for asset_id, positions in asset_positions.items():
        closest_direction = 'neutral'
        closest_distance = float('inf')
        
        for asset_start, asset_end in positions:
            asset_center = (asset_start + asset_end) / 2
            
            for move_start, move_end, indicator, direction in movement_positions:
                move_center = (move_start + move_end) / 2
                distance = abs(asset_center - move_center)
                
                # Prefer movement that comes after asset mention
                if move_center > asset_center:
                    distance *= 0.9
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_direction = direction
        
        if closest_distance < float('inf'):
            asset_movements[asset_id] = closest_direction
    
    return asset_movements


def extract_asset_movement_pairs(text: str) -> Dict[str, str]:
    """
    Extract (asset, direction) pairs by analyzing text structure.
    Handles multi-asset headlines with different movement directions per asset.
    """
    text_lower = text.lower()
    asset_movements = {}
    
    # Get ALL assets mentioned anywhere
    all_assets = detect_assets(text_lower)
    
    # Split on conjunctions for clause-level direction extraction
    split_patterns = [
        r'\s+while\s+',
        r'\s+as\s+',
        r'\s+whereas\s+',
        r'\s+but\s+',
        r'\s+yet\s+',
        r';\s*',
        r',\s+and\s+',
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in split_patterns)
    clauses = re.split(combined_pattern, text_lower, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if c and c.strip() and len(c.strip()) > 3]
    
    if not clauses:
        clauses = [text_lower]
    
    assets_with_directions = set()
    
    # Process each clause
    for clause in clauses:
        clause_assets = set()
        clause_direction = 'neutral'
        
        for asset_id in all_assets:
            keywords = ASSET_KEYWORDS[asset_id]
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r"'?s?\b", clause):
                    clause_assets.add(asset_id)
                    break
        
        clause_direction = infer_direction_from_clause(clause)
        
        for asset in clause_assets:
            if asset not in asset_movements or clause_direction != 'neutral':
                asset_movements[asset] = clause_direction
                assets_with_directions.add(asset)
    
    # Proximity-based refinement for remaining assets
    remaining_assets = all_assets - assets_with_directions
    
    if remaining_assets or len(clauses) == 1:
        proximity_movements = extract_asset_movement_by_proximity(text)
        for asset, direction in proximity_movements.items():
            if asset not in asset_movements or asset_movements[asset] == 'neutral':
                asset_movements[asset] = direction
    
    # Ensure ALL assets have at least neutral direction
    for asset in all_assets:
        if asset not in asset_movements:
            asset_movements[asset] = 'neutral'
    
    return asset_movements


def infer_direction_with_context(text: str, event_type: str,
                                  employment_strength: Optional[str] = None,
                                  mechanisms: Set[str] = None,
                                  asset_type: str = None,
                                  base_direction: str = None) -> str:
    """
    Infer direction considering event type and mechanisms.
    RELAXED VERSION: Less aggressive heuristics, rely more on explicit text.
    """
    if base_direction is not None:
        direction = base_direction
    else:
        direction = 'neutral'
    
    if mechanisms is None:
        mechanisms = set()
    
    # Only apply light context adjustments
    if asset_type == 'vix':
        # VIX has inverse relationship
        text_lower = text.lower()
        vix_explicit = False
        for indicator in (MOVEMENT_INDICATORS['positive'] + MOVEMENT_INDICATORS['strong_positive']):
            if 'vix' in text_lower and indicator in text_lower:
                vix_explicit = True
                break
        
        if not vix_explicit and direction != 'neutral':
            if direction == 'positive':
                direction = 'negative'
            elif direction == 'negative':
                direction = 'positive'
    
    return direction


# ============================================================================
# RELAXED EXTRACTION - No Causality Required
# ============================================================================

def extract_multi_event_relations_relaxed(titles: List[str], df=None) -> Dict:
    """
    RELAXED VERSION: Extract relationships for ANY headline with event/asset mentions.
    NO causal pattern matching required.
    Supports: event-to-event, event-to-asset, and asset-to-asset relationships.
    
    Returns:
        Dictionary with extracted edges (edges with same source, target, relation are merged)
    """
    relations = {
        'events': set(),
        'mechanisms': set(),
        'assets': set(),
        'raw_edges': [],  # Will be aggregated later
        'extraction_stats': {
            'total_headlines': 0,
            'with_events': 0,
            'with_assets': 0,
            'with_both': 0,
            'extracted': 0,
        }
    }
    
    relations['extraction_stats']['total_headlines'] = len(titles)
    
    for idx, title in enumerate(titles):
        if not title or not isinstance(title, str):
            continue
        
        title_lower = title.lower()
        
        # Get metadata
        date = None
        url = None
        if df is not None and idx < len(df):
            if 'date' in df.columns or 'Date' in df.columns:
                col_name = 'date' if 'date' in df.columns else 'Date'
                date = str(df.iloc[idx][col_name]) if pd.notna(df.iloc[idx][col_name]) else None
            if 'url' in df.columns or 'Url' in df.columns:
                col_name = 'url' if 'url' in df.columns else 'Url'
                url = str(df.iloc[idx][col_name]) if pd.notna(df.iloc[idx][col_name]) else None
        
        # Step 1: Detect events
        event_types = detect_event_type(title_lower)
        detected_events = []
        if event_types['employment']:
            detected_events.append('employment')
        if event_types['rate_cut']:
            detected_events.append('rate_cut')
        if event_types['rate_hike']:
            detected_events.append('rate_hike')
        
        has_events = len(detected_events) > 0
        
        # Step 2: Detect assets
        assets = detect_assets(title_lower)
        has_assets = len(assets) > 0
        
        # Skip if neither events nor assets detected
        if not has_events and not has_assets:
            continue
        
        if has_events:
            relations['extraction_stats']['with_events'] += 1
            relations['events'].update(detected_events)
        
        if has_assets:
            relations['extraction_stats']['with_assets'] += 1
            relations['assets'].update(assets)
        
        if has_events and has_assets:
            relations['extraction_stats']['with_both'] += 1
        
        # Step 3: Extract asset-specific directions
        asset_movements = extract_asset_movement_pairs(title)
        
        # Step 4: Create edges - support all relationship types
        relations['extraction_stats']['extracted'] += 1
        
        # Evidence record
        evidence = {
            'title': title,
            'date': date,
            'url': url
        }
        
        # --- Event-to-Event edges ---
        if len(detected_events) > 1:
            for i, event1 in enumerate(detected_events):
                for event2 in detected_events[i+1:]:
                    # Direction is 'neutral' for event-event co-occurrence
                    relations['raw_edges'].append({
                        'source': f'event:{event1}',
                        'source_type': 'event',
                        'target': f'event:{event2}',
                        'target_type': 'event',
                        'relation': 'CO_OCCURRENCE',
                        'evidence': evidence
                    })
        
        # --- Event-to-Asset edges ---
        if has_events and has_assets:
            for event in detected_events:
                for asset in assets:
                    base_direction = asset_movements.get(asset, 'neutral')
                    direction = infer_direction_with_context(
                        title_lower,
                        event,
                        asset_type=asset,
                        base_direction=base_direction
                    )
                    relation = f"{direction.upper()}_IMPACT"
                    
                    relations['raw_edges'].append({
                        'source': f'event:{event}',
                        'source_type': 'event',
                        'target': f'asset:{asset}',
                        'target_type': 'asset',
                        'relation': relation,
                        'evidence': evidence
                    })
        
        # --- Asset-to-Asset edges ---
        if len(assets) > 1:
            assets_list = list(assets)
            for i, asset1 in enumerate(assets_list):
                for asset2 in assets_list[i+1:]:
                    # Direction based on relative movement
                    dir1 = asset_movements.get(asset1, 'neutral')
                    dir2 = asset_movements.get(asset2, 'neutral')
                    
                    if dir1 == dir2:
                        relation = 'POSITIVE_CORRELATION'  # Move together
                    elif dir1 != 'neutral' and dir2 != 'neutral' and dir1 != dir2:
                        relation = 'NEGATIVE_CORRELATION'  # Move opposite
                    else:
                        relation = 'CO_OCCURRENCE'  # Just mentioned together
                    
                    relations['raw_edges'].append({
                        'source': f'asset:{asset1}',
                        'source_type': 'asset',
                        'target': f'asset:{asset2}',
                        'target_type': 'asset',
                        'relation': relation,
                        'evidence': evidence
                    })
    
    # Aggregate edges: merge edges with same (source, target, relation)
    relations['aggregated_edges'] = aggregate_edges(relations['raw_edges'])
    
    return relations


def aggregate_edges(raw_edges: List[Dict]) -> List[Dict]:
    """
    Aggregate edges with same (source, target, relation) into single edge with multiple evidence.
    """
    edge_map = defaultdict(lambda: {'evidence_list': []})
    
    for edge in raw_edges:
        key = (edge['source'], edge['target'], edge['relation'])
        
        if 'source_type' not in edge_map[key]:
            edge_map[key]['source'] = edge['source']
            edge_map[key]['source_type'] = edge['source_type']
            edge_map[key]['target'] = edge['target']
            edge_map[key]['target_type'] = edge['target_type']
            edge_map[key]['relation'] = edge['relation']
        
        edge_map[key]['evidence_list'].append(edge['evidence'])
    
    aggregated = []
    for key, edge_data in edge_map.items():
        aggregated.append({
            'source': edge_data['source'],
            'source_type': edge_data['source_type'],
            'target': edge_data['target'],
            'target_type': edge_data['target_type'],
            'relation': edge_data['relation'],
            'evidence_count': len(edge_data['evidence_list']),
            'evidence_list': edge_data['evidence_list']
        })
    
    # Sort by evidence count descending
    aggregated.sort(key=lambda x: x['evidence_count'], reverse=True)
    
    return aggregated


# ============================================================================
# SUMMARY AND EXPORT FUNCTIONS
# ============================================================================

def summarize_relations_relaxed(relations: Dict) -> Dict:
    """Create comprehensive summary of relationships."""
    summary = {
        'by_event': defaultdict(lambda: {
            'total_edges': 0,
            'to_assets': 0,
            'to_events': 0,
            'assets': defaultdict(int),
        }),
        'by_asset': defaultdict(lambda: {
            'total_edges': 0,
            'from_events': 0,
            'to_assets': 0,
            'events': defaultdict(int),
        }),
        'by_relation_type': defaultdict(int),
        'extraction_stats': relations.get('extraction_stats', {}),
    }
    
    for edge in relations.get('aggregated_edges', []):
        source = edge['source']
        target = edge['target']
        source_type = edge['source_type']
        target_type = edge['target_type']
        relation = edge['relation']
        evidence_count = edge['evidence_count']
        
        # Count by relation type
        summary['by_relation_type'][relation] += 1
        
        # Event stats
        if source_type == 'event':
            event_name = source.replace('event:', '')
            summary['by_event'][event_name]['total_edges'] += 1
            if target_type == 'asset':
                summary['by_event'][event_name]['to_assets'] += 1
                asset_name = target.replace('asset:', '')
                summary['by_event'][event_name]['assets'][asset_name] += 1
            elif target_type == 'event':
                summary['by_event'][event_name]['to_events'] += 1
        
        # Asset stats
        if target_type == 'asset':
            asset_name = target.replace('asset:', '')
            summary['by_asset'][asset_name]['total_edges'] += 1
            if source_type == 'event':
                summary['by_asset'][asset_name]['from_events'] += 1
                event_name = source.replace('event:', '')
                summary['by_asset'][asset_name]['events'][event_name] += 1
        
        if source_type == 'asset':
            asset_name = source.replace('asset:', '')
            summary['by_asset'][asset_name]['total_edges'] += 1
            summary['by_asset'][asset_name]['to_assets'] += 1
    
    return summary


def export_to_json_relaxed(relations: Dict, summary: Dict, output_dir: str = 'output_relaxed'):
    """Export knowledge graph to JSON (relaxed version) with aggregated edges."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    aggregated_edges = relations.get('aggregated_edges', [])
    
    kg = {
        "metadata": {
            "created_at": timestamp,
            "source": "Multi-event financial news extraction (RELAXED - no causality required)",
            "version": "relaxed_v2",
            "description": "KG with aggregated edges. Same (source, target, relation) merged with multiple evidence. Supports event-event, event-asset, asset-asset relationships.",
            "total_nodes": len(relations['events']) + len(relations['assets']),
            "total_edges": len(aggregated_edges),
            "total_evidence": sum(e['evidence_count'] for e in aggregated_edges),
            "extraction_stats": relations.get('extraction_stats', {}),
        },
        "nodes": [],
        "edges": []
    }
    
    # Add event nodes
    for event in sorted(relations['events']):
        kg["nodes"].append({
            "id": f"event:{event}",
            "type": "Event",
            "name": event.replace('_', ' ').title(),
        })
    
    # Add asset nodes
    for asset in sorted(relations['assets']):
        kg["nodes"].append({
            "id": f"asset:{asset}",
            "type": "Asset",
            "name": asset.replace('_', ' ').title(),
        })
    
    # Add aggregated edges
    for idx, edge in enumerate(aggregated_edges, 1):
        kg["edges"].append({
            "id": f"edge:{idx}",
            "source": edge['source'],
            "target": edge['target'],
            "relation": edge['relation'],
            "evidence_count": edge['evidence_count'],
            "evidence": edge['evidence_list']
        })
    
    output_path = os.path.join(output_dir, 'multi_event_causal_kg_relaxed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Knowledge graph exported to: {output_path}")
    print(f"  - {kg['metadata']['total_nodes']} nodes")
    print(f"  - {kg['metadata']['total_edges']} aggregated edges")
    print(f"  - {kg['metadata']['total_evidence']} total evidence pieces")
    
    return output_path


def export_to_csv_relaxed(relations: Dict, output_dir: str = 'output_relaxed'):
    """Export aggregated relationships to CSV (relaxed version)."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_edges = relations.get('aggregated_edges', [])
    
    # Create flattened CSV with one row per edge (evidence as JSON string)
    all_edges = []
    for edge in aggregated_edges:
        all_edges.append({
            'source': edge['source'],
            'source_type': edge['source_type'],
            'target': edge['target'],
            'target_type': edge['target_type'],
            'relation': edge['relation'],
            'evidence_count': edge['evidence_count'],
            'evidence_titles': ' | '.join([e['title'][:80] for e in edge['evidence_list'][:5]]),  # First 5 titles
        })
    
    df = pd.DataFrame(all_edges)
    output_path = os.path.join(output_dir, 'multi_event_causal_relationships_relaxed.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Relationships exported to: {output_path}")
    print(f"  - {len(df)} aggregated edges")
    
    # Also export detailed evidence CSV
    evidence_rows = []
    for edge in aggregated_edges:
        for evidence in edge['evidence_list']:
            evidence_rows.append({
                'source': edge['source'],
                'target': edge['target'],
                'relation': edge['relation'],
                'title': evidence['title'],
                'date': evidence['date'],
                'url': evidence['url']
            })
    
    df_evidence = pd.DataFrame(evidence_rows)
    evidence_path = os.path.join(output_dir, 'multi_event_evidence_detail.csv')
    df_evidence.to_csv(evidence_path, index=False, encoding='utf-8')
    
    print(f"✓ Evidence detail exported to: {evidence_path}")
    print(f"  - {len(df_evidence)} evidence records")
    
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    import os
    
    print("\n" + "=" * 100)
    print("MULTI-EVENT CAUSAL KG EXTRACTION - RELAXED VERSION (Aggregated Edges)")
    print("=" * 100)
    
    output_dir = 'output_relaxed'
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = 'multi_event_mini.csv'
    
    if not os.path.exists(input_file):
        print(f"\n❌ Error: Input file '{input_file}' not found!")
        return
    
    print(f"\n📂 Loading headlines from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ CSV loaded: {len(df)} rows")
        
        title_col = None
        for col in df.columns:
            if 'title' in col.lower() or 'headline' in col.lower():
                title_col = col
                break
        
        if not title_col:
            title_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
        
        print(f"✓ Using column '{title_col}' for headlines")
        
        titles = df[title_col].tolist()
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    print(f"\n🔍 Extracting relationships (NO causality required)...")
    relations = extract_multi_event_relations_relaxed(titles, df)
    
    print(f"\n📊 Generating summary...")
    summary = summarize_relations_relaxed(relations)
    
    # Print results
    print("\n" + "=" * 100)
    print("EXTRACTION RESULTS")
    print("=" * 100)
    
    stats = relations['extraction_stats']
    total = stats['total_headlines']
    print(f"\nHeadline Processing:")
    print(f"  Total headlines: {total}")
    print(f"  With events: {stats['with_events']} ({stats['with_events']/total*100:.1f}%)" if total > 0 else "  With events: 0")
    print(f"  With assets: {stats['with_assets']} ({stats['with_assets']/total*100:.1f}%)" if total > 0 else "  With assets: 0")
    print(f"  With BOTH event+asset: {stats['with_both']} ({stats['with_both']/total*100:.1f}%)" if total > 0 else "  With BOTH: 0")
    print(f"  ✓ Headlines extracted: {stats['extracted']} ({stats['extracted']/total*100:.1f}%)" if total > 0 else "  Headlines extracted: 0")
    
    aggregated_edges = relations.get('aggregated_edges', [])
    raw_edges = relations.get('raw_edges', [])
    
    print(f"\nGraph Structure:")
    print(f"  Events: {len(relations['events'])}")
    print(f"  Assets: {len(relations['assets'])}")
    print(f"  Raw edges (before aggregation): {len(raw_edges)}")
    print(f"  Aggregated edges (unique source-target-relation): {len(aggregated_edges)}")
    
    # Count edge types
    event_event = sum(1 for e in aggregated_edges if e['source_type'] == 'event' and e['target_type'] == 'event')
    event_asset = sum(1 for e in aggregated_edges if e['source_type'] == 'event' and e['target_type'] == 'asset')
    asset_asset = sum(1 for e in aggregated_edges if e['source_type'] == 'asset' and e['target_type'] == 'asset')
    
    print(f"\nEdge Types:")
    print(f"  Event → Event: {event_event}")
    print(f"  Event → Asset: {event_asset}")
    print(f"  Asset → Asset: {asset_asset}")
    
    print(f"\nRelation Breakdown:")
    for relation, count in sorted(summary['by_relation_type'].items(), key=lambda x: -x[1]):
        print(f"  {relation:25} | {count:4} edges")
    
    print(f"\nEvent Stats:")
    for event in sorted(relations['events']):
        data = summary['by_event'][event]
        print(f"  {event:20} | Edges: {data['total_edges']:4} | To assets: {data['to_assets']}, To events: {data['to_events']}")
    
    print(f"\nTop Assets (by edge count):")
    top_assets = sorted(summary['by_asset'].items(), key=lambda x: x[1]['total_edges'], reverse=True)[:10]
    for asset, data in top_assets:
        print(f"  {asset:20} | Edges: {data['total_edges']:4} | From events: {data['from_events']}, To assets: {data['to_assets']}")
    
    print(f"\nTop Aggregated Edges (by evidence count):")
    for edge in aggregated_edges[:10]:
        print(f"  {edge['source']:30} → {edge['target']:30} | {edge['relation']:20} | {edge['evidence_count']} evidence")
    
    # Export results
    print(f"\n💾 Exporting results...")
    csv_path = export_to_csv_relaxed(relations, output_dir)
    json_path = export_to_json_relaxed(relations, summary, output_dir)
    
    print("\n" + "=" * 100)
    print("✓ EXTRACTION COMPLETE!")
    print("=" * 100)
    print(f"\nOutput files in '{output_dir}/':")
    print(f"  1. multi_event_causal_relationships_relaxed.csv  (aggregated edges)")
    print(f"  2. multi_event_evidence_detail.csv               (all evidence records)")
    print(f"  3. multi_event_causal_kg_relaxed.json            (full KG with evidence)")
    if total > 0:
        print(f"\n📊 Coverage: {stats['extracted']/total*100:.1f}% headlines extracted")
        print(f"📊 Aggregation: {len(raw_edges)} raw edges → {len(aggregated_edges)} unique edges")


if __name__ == "__main__":
    main()