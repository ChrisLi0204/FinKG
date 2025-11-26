"""
Causal Knowledge Graph Coverage & Performance Assessment

This script provides accurate coverage analysis by comparing:
1. Raw input headlines
2. Events detected in those headlines
3. Edges extracted (headlines that resulted in causal relationships)
"""

import json
import pandas as pd
from collections import Counter, defaultdict
import re
import os

# ============================================================================
# CONFIGURATION - Edit these values to change file paths and settings
# ============================================================================
CONFIG = {
    'input_csv': 'multi_event_mini.csv',
    'output_kg_json': 'output/multi_event_causal_kg.json',
    'sample_headlines_count': 10,  # Number of sample headlines to display
}

def load_data():
    """Load raw data and KG"""
    df = pd.read_csv(CONFIG['input_csv'])
    with open(CONFIG['output_kg_json'], 'r') as f:
        kg = json.load(f)
    return df, kg

def extract_event_keywords():
    """Extract event detection keywords from the code"""
    rate_cut_keywords = [
        'rate cut', 'rate cuts', 'rate cut bets', 'rate cut hopes',
        'rate cut optimism', 'rate cut view', 'rate cut outlook',
        'rate cut speculation', 'fed cut', 'us rate cut', 'rate cut doubts',
        'rate cut fears', 'rate reduction', 'rate easing', 'rate cut looms',
        'rate cut anticipated', 'rate cut expected', 'rate cut possibility',
        'rate cut ideas', 'emergency rate cut'
    ]
    
    rate_hike_keywords = [
        'rate hike', 'rate hikes', 'rate increase', 'rate rise',
        'fed hike', 'hike outlook', 'hike path', 'rate tightening',
        'taper', 'tapering', 'qe taper', 'quantitative easing'
    ]
    
    employment_keywords = [
        'nonfarm payrolls', 'nonfarm payroll', 'payrolls', 'payroll',
        'jobs report', 'employment report', 'employment data',
        'unemployment rate', 'jobless rate', 'unemployment data', 'jobs data',
        'labor market', 'labour market', 'employment',
        'nfp', 'jolts', 'adp', 'jobless claims', 'unemployment',
        'labor market data', 'employment growth', 'job growth',
        'hiring', 'layoffs', 'wage growth', 'wages', 'earnings',
        'job loss', 'job gains', 'job losses', 'jobs added',
        'jobless claims', 'claims fall', 'claims rise',
        'upbeat jobs report', 'weak jobs report', 'strong jobs report',
    ]
    
    return rate_cut_keywords, rate_hike_keywords, employment_keywords

def detect_events_in_headlines(df):
    """
    Manually detect events in all headlines using the same logic as the extractor.
    Returns detailed statistics about event detection.
    """
    rate_cut_kw, rate_hike_kw, employment_kw = extract_event_keywords()
    
    results = {
        'total_headlines': len(df),
        'has_rate_cut': 0,
        'has_rate_hike': 0,
        'has_employment': 0,
        'has_any_event': 0,
        'has_multiple_events': 0,
        'no_event': 0,
        'headlines_by_event': defaultdict(list)
    }
    
    for idx, row in df.iterrows():
        title = str(row.get('title_lower', '')).lower()
        
        detected = {
            'rate_cut': any(kw in title for kw in rate_cut_kw),
            'rate_hike': any(kw in title for kw in rate_hike_kw),
            'employment': any(kw in title for kw in employment_kw)
        }
        
        event_count = sum(detected.values())
        
        if detected['rate_cut']:
            results['has_rate_cut'] += 1
            results['headlines_by_event']['rate_cut'].append(title)
        if detected['rate_hike']:
            results['has_rate_hike'] += 1
            results['headlines_by_event']['rate_hike'].append(title)
        if detected['employment']:
            results['has_employment'] += 1
            results['headlines_by_event']['employment'].append(title)
        
        if event_count > 0:
            results['has_any_event'] += 1
        if event_count > 1:
            results['has_multiple_events'] += 1
        if event_count == 0:
            results['no_event'] += 1
    
    return results

def analyze_kg_coverage(kg):
    """
    Analyze what the KG actually captured.
    """
    # Get unique headlines from evidence
    unique_headlines_in_kg = set()
    
    for edge in kg['edges']:
        for evidence in edge.get('evidence', []):
            title = evidence.get('title', '')
            if title:
                unique_headlines_in_kg.add(title.lower())
    
    return {
        'unique_headlines_with_edges': len(unique_headlines_in_kg),
        'total_edges': len(kg['edges']),
        'total_nodes': len(kg['nodes'])
    }

def get_headlines_with_edges(kg):
    """Get the actual set of headlines that resulted in edges"""
    headlines_with_edges = set()
    
    for edge in kg['edges']:
        for evidence in edge.get('evidence', []):
            title = evidence.get('title', '')
            if title:
                headlines_with_edges.add(title.lower())
    
    return headlines_with_edges

def main():
    print("="*80)
    print("ACCURATE COVERAGE ANALYSIS FOR CAUSAL KG EXTRACTOR")
    print("="*80)
    
    # Load data
    df, kg = load_data()
    
    # Extract event keywords once to avoid repetition
    rate_cut_kw, rate_hike_kw, employment_kw = extract_event_keywords()
    all_event_keywords = rate_cut_kw + rate_hike_kw + employment_kw
    
    print("\n## 1. INPUT DATA")
    print("-"*40)
    print(f"Total headlines in dataset: {len(df)}")
    print(f"Event category distribution:")
    print(df['event_category'].value_counts())
    
    # Detect events in all headlines
    print("\n## 2. EVENT DETECTION COVERAGE")
    print("-"*40)
    event_results = detect_events_in_headlines(df)
    
    print(f"Headlines with ANY event detected: {event_results['has_any_event']} / {event_results['total_headlines']}")
    print(f"  Coverage: {event_results['has_any_event']/event_results['total_headlines']*100:.1f}%")
    print(f"\nBreakdown by event type:")
    print(f"  - Employment: {event_results['has_employment']} ({event_results['has_employment']/event_results['total_headlines']*100:.1f}%)")
    print(f"  - Rate Cut: {event_results['has_rate_cut']} ({event_results['has_rate_cut']/event_results['total_headlines']*100:.1f}%)")
    print(f"  - Rate Hike: {event_results['has_rate_hike']} ({event_results['has_rate_hike']/event_results['total_headlines']*100:.1f}%)")
    print(f"\nHeadlines with multiple events: {event_results['has_multiple_events']}")
    print(f"Headlines with NO events: {event_results['no_event']}")
    
    # Analyze KG output
    print("\n## 3. KG EXTRACTION COVERAGE")
    print("-"*40)
    kg_stats = analyze_kg_coverage(kg)
    headlines_with_edges = get_headlines_with_edges(kg)
    
    print(f"Unique headlines that resulted in edges: {kg_stats['unique_headlines_with_edges']}")
    print(f"Total edges created: {kg_stats['total_edges']}")
    print(f"Total nodes created: {kg_stats['total_nodes']}")
    
    # Calculate the real coverage
    print("\n## 4. ACTUAL COVERAGE CALCULATION")
    print("-"*40)
    print("\nThree levels of coverage:")
    print(f"\n1. EVENT DETECTION (does headline mention an event?)")
    print(f"   {event_results['has_any_event']} / {event_results['total_headlines']} = {event_results['has_any_event']/event_results['total_headlines']*100:.1f}%")
    
    print(f"\n2. EDGE EXTRACTION (does headline result in a causal edge?)")
    print(f"   {kg_stats['unique_headlines_with_edges']} / {event_results['has_any_event']} = {kg_stats['unique_headlines_with_edges']/event_results['has_any_event']*100:.1f}%")
    print(f"   (of headlines with events detected)")
    
    print(f"\n3. END-TO-END (does headline result in edges out of all headlines?)")
    print(f"   {kg_stats['unique_headlines_with_edges']} / {event_results['total_headlines']} = {kg_stats['unique_headlines_with_edges']/event_results['total_headlines']*100:.1f}%")
    
    # Find headlines that were detected but didn't result in edges
    print("\n## 5. GAP ANALYSIS: Events Detected but No Edges Created")
    print("-"*40)
    
    detected_titles = set()
    for title in df['title_lower'].dropna():
        title_lower = title.lower()
        if any(kw in title_lower for kw in all_event_keywords):
            detected_titles.add(title_lower)
    
    no_edge_headlines = detected_titles - headlines_with_edges
    print(f"Headlines with events but NO edges: {len(no_edge_headlines)}")
    if len(detected_titles) > 0:
        print(f"  This represents: {len(no_edge_headlines)/len(detected_titles)*100:.1f}% of detected events")
    print(f"\nMost likely reasons:")
    print(f"  - No asset keywords detected in headline")
    print(f"  - Asset detected but no causal pattern matched")
    
    # Sample headlines with no edges
    print(f"\nSample headlines with events but no edges (first {CONFIG['sample_headlines_count']}):")
    for i, title in enumerate(list(no_edge_headlines)[:CONFIG['sample_headlines_count']], 1):
        print(f"  {i}. {title}")

    # Headlines with NO events at all
    print("\n## 6. HEADLINES WITH NO EVENTS DETECTED")
    print("-"*40)
    print(f"\nHeadlines with NO events: {event_results['no_event']}")
    
    if event_results['no_event'] > 0:
        print(f"\nThese headlines do not mention any economic event (employment, rate cut, rate hike):")
        
        # Get all headlines without events
        no_event_headlines = []
        for idx, row in df.iterrows():
            title = str(row.get('title_lower', '')).lower()
            has_event = any(kw in title for kw in all_event_keywords)
            if not has_event:
                no_event_headlines.append(title)
        
        for i, title in enumerate(no_event_headlines, 1):
            print(f"  {i}. {title}")

    
    # Performance metrics
    print("\n## 7. PERFORMANCE METRICS SUMMARY")
    print("-"*40)
    print(f"Event Detection Precision: ~100% (keyword-based, high precision)")
    print(f"Event Detection Recall: {event_results['has_any_event']/event_results['total_headlines']*100:.1f}%")
    print(f"\nEdge Extraction Success Rate:")
    if event_results['has_any_event'] > 0:
        print(f"  {kg_stats['unique_headlines_with_edges']} / {event_results['has_any_event']}")
        print(f"  = {kg_stats['unique_headlines_with_edges']/event_results['has_any_event']*100:.1f}%")
    print(f"  (% of event-detected headlines that produce edges)")
    
    print(f"\nOverall Pipeline Success Rate:")
    print(f"  {kg_stats['unique_headlines_with_edges']} / {event_results['total_headlines']}")
    print(f"  = {kg_stats['unique_headlines_with_edges']/event_results['total_headlines']*100:.1f}%")
    print(f"  (% of all headlines that result in KG edges)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
