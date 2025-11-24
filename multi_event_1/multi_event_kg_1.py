"""
Multi-Event Causal Knowledge Graph Extraction for Financial Markets
====================================================================

This module extracts causal relationships between multiple economic events 
(US rate cuts, Employment data) and financial assets from news headlines.
It includes a Mechanism/Context layer to capture market sentiment, expectations,
and transmission channels.

Enhanced multi-event architecture with 5 layers:
- Layer 0: Provenance (headlines, sources, dates)
- Layer 1: Events (MonetaryPolicy, LaborMarket)
- Layer 2: Mechanisms/Context (expectations, sentiment, channels)
- Layer 3: Assets/Markets (FX, Equities, Commodities, Bonds)
- Layer 4: Outcomes/Metrics (optional, for future quantitative validation)

"""

import re
import json
from collections import defaultdict
from datetime import datetime
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional


# ============================================================================
# ASSET KEYWORDS DICTIONARY (Extended with improved coverage)
# ============================================================================
ASSET_KEYWORDS = {
    # Precious metals
    'gold': ['gold', 'bullion', 'xau'],
    'silver': ['silver', 'xag'],
    
    # Major currencies
    'dollar': ['dollar', 'usd', 'greenback', 'dlr', 'dxy', 'us dollar'],
    'euro': ['euro', 'eur'],
    'yen': ['yen', 'jpy'],
    'pound': ['pound', 'gbp', 'sterling', 'cable'],
    'peso': ['peso', 'mxn'],
    'real': ['real', 'brl'],
    'canadian_dollar': ['canadian dollar', 'c$', 'loonie', 'cad'],
    'aussie': ['aussie', 'australian dollar', 'aud'],
    'franc': ['franc', 'chf', 'swiss franc'],
    'krona': ['krona', 'sek'],
    'rand': ['rand', 'zar', 'south african rand'],
    
    # General equity markets
    'stocks': ['stocks', 'shares', 'equity', 'equities', 'wall street', 'wall st'],
    'reit': ['reit', 'reits', 'real estate'],
    
    # Major indices
    'sp500': ['s&p 500', 's&p', 'spx', 'sp500'],
    'dow': ['dow', 'djia', 'dow jones'],
    'nasdaq': ['nasdaq'],
    'nikkei': ['nikkei'],
    'ftse': ['ftse'],
    
    # Regional equity markets
    'seoul_stocks': ['seoul shares', 'seoul stocks'],
    'brazil_stocks': ['brazil stocks', 'brazilian stocks', 'bovespa', 'brazil'],
    'mexico_stocks': ['mexico stocks', 'mexican stocks', 'mexico'],
    'hk_stocks': ['hk shares', 'hong kong shares', 'hk stocks', 'hang seng'],
    'china_stocks': ['china stocks', 'shanghai', 'shenzhen'],
    'india_stocks': ['india stocks', 'sensex', 'nse'],
    
    'asia_stocks': [
        'asian shares', 'asian stocks', 'asia shares', 'asia stocks',
        'asia up', 'asia down', 'asia gains', 'asia rises', 'asia falls',
        'se asia', 'southeast asia'
    ],
    
    'europe_stocks': ['europe shares', 'europe stocks', 'european shares', 'european stocks'],
    'uk_stocks': ['uk shares', 'uk stocks', 'british shares'],
    'canada_stocks': ['canadian stocks', 'tsx'],
    
    # Fixed income
    'bonds': ['bonds', 'treasuries', 'debt', 'treasury', 'gilt', 'bund', 'bond market'],
    'ust2y': ['2-year', '2y', '2-yr', 'two-year yield'],
    'ust10y': ['10-year', '10y', '10-yr', 'ten-year yield'],
    'ust30y': ['30-year', '30y', 'thirty-year yield'],
    'credit_spreads': ['credit spreads', 'credit default swaps', 'cds'],
    
    # Other asset classes
    'emerging_markets': ['emerging markets', 'emerging debt', 'em', 'latam', 'em fx'],
    'yields': ['yields', 'yield', 'treasury yield', 'yield curve'],
    'currencies': ['currencies', 'currency', 'forex', 'fx'],
    'commodities': ['commodities', 'commodity'],
    'crude': ['crude', 'oil', 'wti', 'brent', 'nymex'],
    'copper': ['copper', 'copper futures'],
    'gasoline': ['gasoline', 'rbob'],
    
    # Sectors
    'financials': ['financials', 'banks', 'banking', 'financial sector', 'financial stocks'],
    'tech': ['tech', 'technology', 'technology sector', 'semiconductor', 'chip stocks'],
    'retail': ['retail', 'retail sector', 'retail stocks', 'department stores'],
    'energy': ['energy', 'energy sector', 'oil stocks', 'energy stocks'],
    'healthcare': ['healthcare', 'pharma', 'biotech'],
    'consumer': ['consumer', 'consumer staples', 'consumer discretionary'],
    
    # Small Caps
    'small_caps': ['small cap', 'smallcap', 'small-cap', 'russell 2000', 'russell', 'small caps'],
    
    # Cryptocurrency
    'crypto': ['crypto', 'cryptocurrency', 'bitcoin', 'btc', 'digital currency', 'digital asset'],
}

# Asset type mapping
ASSET_TYPE_MAP = {
    'gold': 'Commodity',
    'silver': 'Commodity',
    'commodities': 'Commodity',
    'crude': 'Commodity',
    'dollar': 'Currency',
    'euro': 'Currency',
    'yen': 'Currency',
    'peso': 'Currency',
    'real': 'Currency',
    'canadian_dollar': 'Currency',
    'aussie': 'Currency',
    'stocks': 'EquityMarket',
    'sp500': 'EquityIndex',
    'dow': 'EquityIndex',
    'nasdaq': 'EquityIndex',
    'nikkei': 'EquityIndex',
    'ftse': 'EquityIndex',
    'seoul_stocks': 'RegionalEquity',
    'brazil_stocks': 'RegionalEquity',
    'mexico_stocks': 'RegionalEquity',
    'hk_stocks': 'RegionalEquity',
    'asia_stocks': 'RegionalEquity',
    'europe_stocks': 'RegionalEquity',
    'emerging_markets': 'AssetClass',
    'bonds': 'FixedIncome',
    'yields': 'FixedIncome',
    'currencies': 'AssetClass',
    'commodities': 'AssetClass',
    'financials': 'EquitySector',
    'tech': 'EquitySector',
    'retail': 'EquitySector',
    'small_caps': 'EquityIndex',
    'crypto': 'DigitalAsset',
}

# Asset display names
ASSET_DISPLAY_NAMES = {
    'gold': 'Gold',
    'silver': 'Silver',
    'crude': 'Crude Oil',
    'dollar': 'US Dollar',
    'euro': 'Euro',
    'yen': 'Japanese Yen',
    'peso': 'Mexican Peso',
    'real': 'Brazilian Real',
    'canadian_dollar': 'Canadian Dollar',
    'aussie': 'Australian Dollar',
    'stocks': 'Global Stocks',
    'sp500': 'S&P 500',
    'dow': 'Dow Jones Industrial Average',
    'nasdaq': 'Nasdaq Composite',
    'nikkei': 'Nikkei 225',
    'ftse': 'FTSE 100',
    'seoul_stocks': 'Seoul Stock Market',
    'brazil_stocks': 'Brazilian Stocks',
    'mexico_stocks': 'Mexican Stocks',
    'hk_stocks': 'Hong Kong Stocks',
    'asia_stocks': 'Asian Stocks',
    'europe_stocks': 'European Stocks',
    'emerging_markets': 'Emerging Markets',
    'bonds': 'Bonds/Treasuries',
    'yields': 'Treasury Yields',
    'currencies': 'Foreign Exchange',
    'commodities': 'Commodities',
    'financials': 'Financial Sector',
    'tech': 'Technology Sector',
    'retail': 'Retail Sector',
    'small_caps': 'Small Cap Stocks (Russell 2000)',
    'crypto': 'Cryptocurrency',
}


# ============================================================================
# EVENT KEYWORDS
# ============================================================================

# Monetary Policy Events
RATE_CUT_KEYWORDS = [
    'rate cut', 'rate cuts', 'rate cut bets', 'rate cut hopes',
    'rate cut optimism', 'rate cut view', 'rate cut outlook',
    'rate cut speculation', 'fed cut', 'us rate cut', 'rate cut doubts',
    'rate cut fears', 'rate reduction', 'rate easing', 'rate cut looms',
    'rate cut anticipated', 'rate cut expected', 'rate cut possibility',
    'rate cut ideas', 'emergency rate cut'
]

RATE_HIKE_KEYWORDS = [
    'rate hike', 'rate hikes', 'rate increase', 'rate rise',
    'fed hike', 'hike outlook', 'hike path', 'rate tightening',
    'taper', 'tapering', 'qe taper', 'quantitative easing'
]

# Labor Market Events
EMPLOYMENT_KEYWORDS = [
    'nonfarm payrolls', 'nonfarm payroll', 'payrolls', 'payroll',
    'jobs report', 'employment report', 'employment data',
    'unemployment rate', 'jobless rate', 'unemployment data', 'jobs data',
    'labor market', 'labour market', 'jobs data', 'employment',
    'nfp', 'jolts', 'adp', 'jobless claims', 'unemployment',
    'labor market data', 'employment growth', 'job growth',
    'hiring', 'layoffs', 'wage growth', 'wages', 'earnings',
    'job loss', 'job gains', 'job losses', 'jobs added',
    'unemployment figure', 'employment situation', 'establishment data',
    'initial jobless claims', 'continuing claims', 'claims data',
    'labor force participation', 'participation rate',
    'underemployment', 'discouraged workers',
    'job openings', 'quit rate', 'quits data',
    'staffing', 'labor supply', 'worker shortage'
]


# ============================================================================
# MECHANISM/CONTEXT KEYWORDS
# ============================================================================

MECHANISM_KEYWORDS = {
    # Expectation Timing - ENHANCED
    'ahead_of_jobs': {
        'id': 'mech:ahead_of_jobs_report',
        'name': 'Ahead of Jobs Report',
        'type': 'Expectation_Timing',
        'patterns': [
            r'ahead of.*?(jobs report|employment data|nonfarm payrolls|nfp|unemployment)',
            r'before.*?(jobs report|employment data|nonfarm payrolls|payroll)',
            r'(awaits?|awaiting|eyes on).*?(jobs report|employment data|payroll)',
            r'(jobs report|employment data|payroll).*(looms|on tap|in focus|due|expected)',
            r'watch.*?(jobs report|employment data|payroll)',
            r'ahead.*?(labor market|employment)',
        ]
    },
    'after_jobs': {
        'id': 'mech:after_jobs_report',
        'name': 'After Jobs Report',
        'type': 'Expectation_Timing',
        'patterns': [
            r'after.*?(jobs report|employment data|nonfarm payrolls|nfp)',
            r'on.*?(jobs report|employment data|nonfarm payrolls)',
            r'following.*?(jobs report|employment data|payroll)',
            r'post-?(jobs report|employment)',
        ]
    },
    
    # Rate Cut Expectations - ENHANCED
    'rate_cut_bets': {
        'id': 'mech:rate_cut_bets',
        'name': 'Rate Cut Expectations (Positive)',
        'type': 'Policy_Expectation',
        'patterns': [
            r'rate cut (hopes?|bets?|speculation|optimism|view|outlook|ideas|odds?)',
            r'(hopes?|bets?|expects?|sees?|pins?)\s+(for|on).*?rate cut',
            r'rate cut expectations?',
            r'(cut.*?chances?|chances?.*?cut)',
            r'(trimmed?|pared?|cut|slash).*?rate (hike|increase)',
        ]
    },
    'rate_cut_fears': {
        'id': 'mech:rate_cut_fears',
        'name': 'Rate Cut Concerns (Negative)',
        'type': 'Policy_Expectation',
        'patterns': [
            r'rate cut (doubts?|fears?|concerns?|skepticism|dim)',
            r'(reduce|dim|fade).*?(cut.*?chances?|rate cut)',
        ]
    },
    
    # Rate Hike Expectations - NEW
    'rate_hike_bets': {
        'id': 'mech:rate_hike_bets',
        'name': 'Rate Hike Expectations',
        'type': 'Policy_Expectation',
        'patterns': [
            r'rate hike (hopes?|bets?|odds?|expectations?)',
            r'hike (odds?|chances?|expectations?)',
            r'(boost|raise|lift).*?(hike|tightening)',
        ]
    },
    
    # Policy Repricing - ENHANCED
    'hawkish_repricing': {
        'id': 'mech:hawkish_repricing',
        'name': 'Hawkish Policy Repricing',
        'type': 'Policy_Repricing',
        'patterns': [
            r'(sparks?|ignites?|fuels?|stokes?|renews?).*(inflation|rate hike|hike|tightening)',
            r'(cements?|keeps?|supports?|puts).*on.*hike (path|track)',
            r'hike (jitters|worries|concerns|fears)',
            r'(reduce|dim|pare|fade).*(cut.*?chances?|cut.*?odds?)',
            r'(keeps?|supports?|bolsters?).*hike',
        ]
    },
    'dovish_repricing': {
        'id': 'mech:dovish_repricing',
        'name': 'Dovish Policy Repricing',
        'type': 'Policy_Repricing',
        'patterns': [
            r'(boost|lift|raise|increase|sees?|spurs?).*(cut.*?chances?|cut.*?odds?|cut.*?bets)',
            r'path to.*?cuts? (clearer|stronger)',
            r'(dovish|easing|accommodative).*(tone|tilt|pivot|shift)',
            r'(supports?|fuels?).*fed rate cut',
        ]
    },
    
    # Labor Market State - ENHANCED
    'tight_labor_market': {
        'id': 'mech:tight_labor_market',
        'name': 'Tight Labor Market',
        'type': 'Labor_State',
        'patterns': [
            r'tight(ening)?\s+(labor|labour)\s+market',
            r'(labor|labour)\s+market\s+tighten(s|ing)?',
            r'full\s+(strength|employment)',
            r'robust\s+(labor|labour)\s+market',
            r'resilient\s+(labor|labour)\s+market',
            r'(strong|resilient|robust).*hiring',
        ]
    },
    'weak_labor_market': {
        'id': 'mech:weak_labor_market',
        'name': 'Weak Labor Market',
        'type': 'Labor_State',
        'patterns': [
            r'weak(ening)?\s+(labor|labour)\s+market',
            r'soft\s+(labor|labour)\s+market',
            r'cooling\s+(labor|labour)\s+market',
            r'(labor|labour)\s+market\s+(struggles?|woes|concerns)',
            r'ebbing.*?momentum',
            r'slowing.*?(labor|labour|hiring)',
            r'(disappointing|dismal|gloomy)\s+(labor|labour|employment)',
        ]
    },
    
    # New: Unemployment State
    'unemployment_low': {
        'id': 'mech:unemployment_low',
        'name': 'Low Unemployment Rate',
        'type': 'Labor_State',
        'patterns': [
            r'(unemployment|jobless).*?(falls?|drops?|declines?|hits?.*?(low|year|decade))',
            r'(unemployment|jobless).*?(3\.[0-9]|4\.[0-9]|5\.[0-9])\%',
        ]
    },
    'unemployment_high': {
        'id': 'mech:unemployment_high',
        'name': 'High Unemployment Rate',
        'type': 'Labor_State',
        'patterns': [
            r'(unemployment|jobless).*?(rises?|climbs?|jumps?|hits?.*?(high|peak))',
            r'(unemployment|jobless).*?(8\.|9\.|10\.|11\.)[0-9]\%',
        ]
    },
    
    # New: Wage Dynamics
    'wage_pressure': {
        'id': 'mech:wage_pressure',
        'name': 'Wage Pressure/Growth',
        'type': 'Macro_Channel',
        'patterns': [
            r'wage(s)?.*?(growth|pressure|rise|increase|rise)',
            r'(rising|strong).*?wage',
            r'(pay|salary|compensation).*?(rise|increase)',
        ]
    },
}


# ============================================================================
# MOVEMENT INDICATORS
# ============================================================================
MOVEMENT_INDICATORS = {
    'strong_positive': [
        'surge', 'soar', 'rally', 'jump', 'spike', 'boom', 'explode', 'rocket',
        'blowout', 'smashing', 'crushing', 'stellar', 'bullish', 'beat',
        'crushed expectations', 'stronger than expected'
    ],
    'positive': [
        'gain', 'rise', 'climb', 'advance', 'improve', 'boost', 'up', 'higher',
        'support', 'lift', 'strengthen', 'firm', 'extend gains', 'rebound',
        'better than', 'outperform', 'exceed', 'cheer', 'churred'
    ],
    'weak_positive': [
        'edge up', 'inch up', 'recover', 'rebound', 'trim losses', 'pare losses',
        'stabilize', 'steady', 'hold', 'support', 'moderate gains'
    ],
    
    'strong_negative': [
        'plunge', 'crash', 'collapse', 'tumble', 'slammed', 'crushed', 'sink',
        'tanked', 'hammered', 'battered', 'selloff', 'massacre', 'bloodbath',
        'dismal', 'gloomy', 'grim', 'bleak'
    ],
    'negative': [
        'fall', 'drop', 'decline', 'slip', 'slide', 'down', 'lower', 'weaken',
        'bruised', 'wounded', 'hit', 'off', 'dip', 'ease', 'soften', 'pressure',
        'bearish', 'miss', 'disappoint', 'worse than', 'underperform'
    ],
    'weak_negative': [
        'edge down', 'pare gains', 'retreat', 'modest decline', 'consolidate',
        'pullback', 'correction', 'cautious', 'wary'
    ],
    
    'neutral': [
        'flat', 'steady', 'stable', 'unchanged', 'mixed', 'choppy', 'volatile',
        'little changed', 'narrowly', 'range-bound', 'muted', 'subdued'
    ]
}


# ============================================================================
# CAUSAL PATTERN DEFINITIONS
# ============================================================================

CAUSAL_PATTERNS = [
    # High Priority: Explicit Causal Language
    {
        'name': 'explicit_on_after',
        'pattern': r'(on|after|following|amid)\s+.*?(rate cut|fed cut|jobs report|employment data|nonfarm payrolls)',
        'requires_movement': True,
        'priority': 10
    },
    {
        'name': 'explicit_due_to',
        'pattern': r'(due to|thanks to|because of|as a result of)\s+.*?(rate cut|fed cut|jobs report|employment)',
        'requires_movement': True,
        'priority': 10
    },
    {
        'name': 'event_causes_asset',
        'pattern': r'(rate cut|fed cut|jobs report|employment data)\s+(boosts?|sends?|drives?|pushes?|lifts?|supports?|weighs on|hurts?|hits?|pressures?)',
        'priority': 10
    },
    
    # Medium-High Priority: Expectation/Anticipation Patterns
    {
        'name': 'movement_as_expectation',
        'pattern': r'(as|while)\s+.*?(rate cut|jobs report|employment)\s+(hopes?|bets?|expectations?|optimism|speculation|doubts?|fears?)',
        'requires_movement': True,
        'priority': 9
    },
    {
        'name': 'event_quality_reaction',
        'pattern': r'(strong|weak|disappointing|better|worse|blowout|dismal)\s+.*?(jobs report|employment).*?(send|lift|weigh|boost|hurt)\s+\w+\s+(higher|lower)',
        'priority': 8
    },
    {
        'name': 'movement_before',
        'pattern': r'(before|ahead of|awaiting?|anticipating?)\s+.*?(rate cut|jobs report|employment data)',
        'requires_movement': True,
        'priority': 8
    },
    
    # Medium Priority: Contextual Patterns
    {
        'name': 'event_sends_direction',
        'pattern': r'(rate cut|jobs report|employment)\s+(hopes?|bets?|data)?\s+(send|push|drive|lift)\s+\w+\s+(higher|lower|up|down)',
        'priority': 7
    },
    {
        'name': 'asset_move_ahead_event',
        'pattern': r'(stocks?|dollar|bond|gold|market|yen|euro|crude)\s+(rise|fall|rally|plunge|surge|slide).*?(ahead of|before|as|awaiting)\s+.*?(jobs report|employment|rate cut)',
        'priority': 7
    },
    {
        'name': 'policy_expectations_impact',
        'pattern': r'(rate cut|rate hike|fed policy|monetary policy)\s+(hopes?|bets?|speculation|fears?)\s+(lift|weigh|boost|hit|hurt)\s+\w+\s+(higher|lower)',
        'priority': 7
    },
    {
        'name': 'direct_asset_event_link',
        'pattern': r'(dollar|stocks?|bond|gold|yield|crude|yen)\s+(strength|weakness|gain|loss).*?(on|amid|after)\s+(rate cut|jobs report)',
        'priority': 6
    },
]


# ============================================================================
# CORE EXTRACTION FUNCTIONS
# ============================================================================

def detect_event_type(text: str) -> Dict[str, bool]:
    """
    Detect which event types are mentioned in the text.
    Returns dictionary with event type flags and state information.
    """
    text_lower = text.lower()
    
    events = {
        'rate_cut': any(keyword in text_lower for keyword in RATE_CUT_KEYWORDS),
        'rate_hike': any(keyword in text_lower for keyword in RATE_HIKE_KEYWORDS),
        'employment': any(keyword in text_lower for keyword in EMPLOYMENT_KEYWORDS)
    }
    
    return events


def detect_mechanisms(text: str) -> Set[str]:
    """
    Detect mechanism/context nodes mentioned in the text.
    Returns set of mechanism IDs.
    """
    detected = set()
    
    for mech_key, mech_config in MECHANISM_KEYWORDS.items():
        for pattern in mech_config['patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                detected.add(mech_config['id'])
                break
    
    return detected


def detect_employment_strength(text: str) -> Optional[str]:
    """
    Detect whether employment data is characterized as strong/weak/mixed.
    
    Returns:
        'strong', 'weak', 'mixed', or None
    """
    text_lower = text.lower()
    
    strong_indicators = [
        'strong.*job', 'robust.*job', 'blowout.*job', 'solid.*job',
        'beat.*expect', 'exceed.*expect', 'better.*than.*expect',
        'surprise.*job', 'stunning.*job', 'crush', 'smashing'
    ]
    
    weak_indicators = [
        'weak.*job', 'soft.*job', 'tepid.*job', 'disappointing.*job',
        'dismal.*job', 'grim.*job', 'miss.*expect', 'below.*expect',
        'worse.*than.*expect', 'slump', 'faltered'
    ]
    
    mixed_indicators = ['mixed.*job', 'mixed.*employment', 'mixed.*labor']
    
    if any(re.search(pattern, text_lower) for pattern in strong_indicators):
        return 'strong'
    elif any(re.search(pattern, text_lower) for pattern in weak_indicators):
        return 'weak'
    elif any(re.search(pattern, text_lower) for pattern in mixed_indicators):
        return 'mixed'
    
    return None


# ============================================================================
# ASSET DETECTION
# ============================================================================

def detect_assets(text: str) -> Set[str]:
    """
    Detect all asset types mentioned in the text.
    Returns set of asset type identifiers.
    """
    detected = set()
    
    for asset_type, keywords in ASSET_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                detected.add(asset_type)
                break
    
    return detected


def infer_direction_from_movement(text: str) -> str:
    """
    Infer direction based on movement indicators in the text.
    """
    # Check negative first (often more explicit)
    for strength in ['strong_negative', 'negative', 'weak_negative']:
        for indicator in MOVEMENT_INDICATORS[strength]:
            if indicator in text:
                return 'negative'
    
    # Check positive
    for strength in ['strong_positive', 'positive', 'weak_positive']:
        for indicator in MOVEMENT_INDICATORS[strength]:
            if indicator in text:
                return 'positive'
    
    # Check neutral
    for indicator in MOVEMENT_INDICATORS['neutral']:
        if indicator in text:
            return 'neutral'
    
    return 'neutral'


def infer_direction_with_context(text: str, event_type: str, 
                                  employment_strength: Optional[str] = None,
                                  mechanisms: Set[str] = None) -> str:
    """
    Infer direction considering event type and mechanisms.
    
    Employment heuristics:
    - Weak jobs + rate cut bets â†’ dovish â†’ USD down, bonds up, gold up, stocks mixed-positive
    - Strong jobs + hike worries â†’ hawkish â†’ USD up, bonds down, gold down, stocks mixed-negative
    """
    # Start with movement-based direction
    base_direction = infer_direction_from_movement(text)
    
    if mechanisms is None:
        mechanisms = set()
    
    # Adjust based on employment strength and mechanisms
    if event_type == 'employment' and employment_strength:
        if employment_strength == 'weak':
            if 'mech:rate_cut_bets' in mechanisms or 'mech:dovish_repricing' in mechanisms:
                # Weak jobs + dovish â†’ generally positive for bonds/gold, negative for USD
                if any(word in text.lower() for word in ['dollar', 'usd', 'greenback']):
                    return 'negative'
                elif any(word in text.lower() for word in ['bond', 'treasur', 'gold']):
                    return 'positive'
        
        elif employment_strength == 'strong':
            if 'mech:rate_hike_worries' in mechanisms or 'mech:hawkish_repricing' in mechanisms:
                # Strong jobs + hawkish â†’ generally positive for USD, negative for bonds/gold
                if any(word in text.lower() for word in ['dollar', 'usd', 'greenback']):
                    return 'positive'
                elif any(word in text.lower() for word in ['bond', 'treasur', 'gold']):
                    return 'negative'
    
    return base_direction


def match_causal_pattern(text: str, asset_type: str) -> Optional[Tuple[str, str]]:
    """
    Match causal patterns in the text for a given asset.
    Returns (pattern_name, direction) or None.
    """
    sorted_patterns = sorted(CAUSAL_PATTERNS, key=lambda x: x['priority'], reverse=True)
    
    for pattern_config in sorted_patterns:
        if re.search(pattern_config['pattern'], text, re.IGNORECASE):
            direction = infer_direction_from_movement(text)
            return (pattern_config['name'], direction)
    
    # Fallback
    return ('general_context', infer_direction_from_movement(text))


def extract_multi_event_relations(titles: List[str], df: pd.DataFrame = None) -> Dict:
    """
    Extract causal relationships from news headlines with multi-event support.
    
    Returns:
        Dictionary containing:
        - events: Set of event nodes
        - mechanisms: Set of mechanism nodes
        - assets: Set of asset nodes
        - event_edges: List of (event, mechanism/asset, ...) tuples
        - mechanism_edges: List of (mechanism, asset, ...) tuples
    """
    relations = {
        'events': set(),
        'mechanisms': set(),
        'assets': set(),
        'event_edges': [],  # Event -> Mechanism or Event -> Asset
        'mechanism_edges': [],  # Mechanism -> Asset
    }
    
    for idx, title in enumerate(titles):
        title_lower = title.lower()
        
        # Detect events
        event_types = detect_event_type(title_lower)
        if not any(event_types.values()):
            continue
        
        # Extract metadata
        date = None
        url = None
        if df is not None:
            try:
                date = df.iloc[idx].get('Date')
                url = df.iloc[idx].get('Url')
            except:
                pass
        
        # Detect mechanisms and assets
        mechanisms = detect_mechanisms(title_lower)
        assets = detect_assets(title_lower)
        
        if not assets:
            continue
        
        # Track employment strength for context-aware direction inference
        employment_strength = None
        primary_event = None
        
        if event_types['employment']:
            employment_strength = detect_employment_strength(title_lower)
            primary_event = 'employment'
            relations['events'].add('employment')
        
        if event_types['rate_cut']:
            primary_event = 'rate_cut'
            relations['events'].add('rate_cut')
        
        if event_types['rate_hike']:
            primary_event = 'rate_hike'
            relations['events'].add('rate_hike')
        
        # Add to sets
        relations['mechanisms'].update(mechanisms)
        relations['assets'].update(assets)
        
        # Build edges with improved direction inference
        event_type_for_context = 'employment' if event_types['employment'] else 'monetary_policy'
        direction = infer_direction_with_context(
            title_lower, event_type_for_context, employment_strength, mechanisms
        )
        
        for asset in assets:
            if mechanisms:
                # Path: Event -> Mechanism -> Asset
                for mechanism in mechanisms:
                    # Event -> Mechanism edge
                    result = match_causal_pattern(title_lower, asset)
                    pattern_name = result[0] if result else 'general_context'
                    
                    relations['event_edges'].append((
                        primary_event,
                        mechanism,
                        'mechanism',
                        direction,
                        title,
                        pattern_name,
                        date,
                        url
                    ))
                    
                    # Mechanism -> Asset edges
                    relations['mechanism_edges'].append((
                        mechanism,
                        asset,
                        direction,
                        title,
                        pattern_name,
                        date,
                        url
                    ))
            else:
                # Direct path: Event -> Asset
                result = match_causal_pattern(title_lower, asset)
                if result:
                    pattern_name, inferred_direction = result
                    # Use context-aware direction if available, otherwise use pattern-inferred
                    final_direction = direction if direction != 'neutral' else inferred_direction
                    
                    relations['event_edges'].append((
                        primary_event,
                        asset,
                        'asset',
                        final_direction,
                        title,
                        pattern_name,
                        date,
                        url
                    ))
    
    return relations


# ============================================================================
# SUMMARY AND ANALYSIS FUNCTIONS
# ============================================================================

def summarize_relations(relations: Dict) -> Dict:
    """
    Create comprehensive summary of relationships.
    """
    summary = {
        'by_event': defaultdict(lambda: {
            'total_mentions': 0,
            'mechanisms': defaultdict(int),
            'assets': defaultdict(int),
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }),
        'by_mechanism': defaultdict(lambda: {
            'total_mentions': 0,
            'assets': defaultdict(int),
            'polarity': defaultdict(int),
        }),
        'by_asset': defaultdict(lambda: {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0,
            'events': defaultdict(int),
            'mechanisms': defaultdict(int),
        }),
    }
    
    # Summarize event edges
    for edge in relations['event_edges']:
        event, target, target_type, direction, title, pattern, date, url = edge
        summary['by_event'][event]['total_mentions'] += 1
        summary['by_event'][event][direction] += 1
        
        if target_type == 'mechanism':
            summary['by_event'][event]['mechanisms'][target] += 1
        else:  # asset
            summary['by_event'][event]['assets'][target] += 1
            summary['by_asset'][target][direction] += 1
            summary['by_asset'][target]['total'] += 1
            summary['by_asset'][target]['events'][event] += 1
    
    # Summarize mechanism edges
    for edge in relations['mechanism_edges']:
        mechanism, asset, direction, title, pattern, date, url = edge
        summary['by_mechanism'][mechanism]['total_mentions'] += 1
        summary['by_mechanism'][mechanism]['assets'][asset] += 1
        summary['by_mechanism'][mechanism]['polarity'][direction] += 1
        
        summary['by_asset'][asset][direction] += 1
        summary['by_asset'][asset]['total'] += 1
        summary['by_asset'][asset]['mechanisms'][mechanism] += 1
    
    return summary


def print_extraction_results(relations: Dict, summary: Dict):
    """
    Print formatted extraction results.
    """
    print("=" * 100)
    print("MULTI-EVENT CAUSAL KNOWLEDGE GRAPH EXTRACTION RESULTS")
    print("=" * 100)
    
    print(f"\n{'Event Nodes:':<25} {sorted(list(relations['events']))}")
    print(f"{'Mechanism Nodes:':<25} {len(relations['mechanisms'])}")
    print(f"{'Asset Nodes:':<25} {len(relations['assets'])}")
    print(f"{'Event Edges:':<25} {len(relations['event_edges'])}")
    print(f"{'Mechanism Edges:':<25} {len(relations['mechanism_edges'])}")
    print(f"{'Total Relationships:':<25} {len(relations['event_edges']) + len(relations['mechanism_edges'])}")
    
    # Event summary
    print("\n" + "=" * 100)
    print("EVENT SUMMARY")
    print("=" * 100)
    for event, data in summary['by_event'].items():
        print(f"\n{event.upper().replace('_', ' ')}")
        print(f"  Total mentions: {data['total_mentions']}")
        print(f"  Top mechanisms: {dict(sorted(data['mechanisms'].items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"  Top assets: {dict(sorted(data['assets'].items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    # Mechanism summary
    print("\n" + "=" * 100)
    print("MECHANISM SUMMARY (Top 10)")
    print("=" * 100)
    sorted_mechs = sorted(summary['by_mechanism'].items(), key=lambda x: x[1]['total_mentions'], reverse=True)[:10]
    for mech, data in sorted_mechs:
        mech_name = next((m['name'] for m in MECHANISM_KEYWORDS.values() if m['id'] == mech), mech)
        print(f"\n{mech_name} ({mech})")
        print(f"  Mentions: {data['total_mentions']}")
        print(f"  Polarity: {dict(data['polarity'])}")
        print(f"  Top assets: {dict(sorted(data['assets'].items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    # Asset summary
    print("\n" + "=" * 100)
    print("ASSET SUMMARY")
    print("=" * 100)
    print(f"\n{'Asset':<25} {'Pos':>8} {'Neg':>8} {'Neu':>8} {'Total':>8} {'Dominant':<12}")
    print("-" * 100)
    
    for asset in sorted(summary['by_asset'].keys()):
        data = summary['by_asset'][asset]
        pos = data['positive']
        neg = data['negative']
        neu = data['neutral']
        total = data['total']
        
        if pos > neg and pos > neu:
            dominant = "Positive"
        elif neg > pos and neg > neu:
            dominant = "Negative"
        elif neu > pos and neu > neg:
            dominant = "Neutral"
        else:
            dominant = "Mixed"
        
        display_name = ASSET_DISPLAY_NAMES.get(asset, asset.replace('_', ' ').title())
        print(f"{display_name:<25} {pos:>8} {neg:>8} {neu:>8} {total:>8} {dominant:<12}")


# ============================================================================
# JSON EXPORT FUNCTIONS
# ============================================================================

def build_multi_event_knowledge_graph(relations: Dict, summary: Dict) -> Dict:
    """
    Build a structured JSON knowledge graph with 5-layer architecture.
    """
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    kg = {
        "metadata": {
            "created_at": timestamp,
            "source": "Multi-event financial news headline extraction",
            "event_types": ["Monetary Policy", "Labor Market"],
            "extraction_method": "Rule-based pattern matching with mechanism layer",
            "architecture": "5-layer: Provenance -> Event -> Mechanism -> Asset -> Outcome",
            "total_nodes": 0,
            "total_edges": 0,
            "description": (
                "Multi-event causal knowledge graph extracted from financial news headlines "
                "covering US rate cuts and employment data, with explicit mechanism/context layer"
            )
        },
        "nodes": [],
        "edges": []
    }
    
    # Layer 1: Event Nodes
    event_display = {
        'monetary_policy': 'US Monetary Policy (Rate Cuts/Hikes)',
        'labor_market': 'US Labor Market (Employment Data)'
    }
    
    for event_id in sorted(relations['events']):
        event_node = {
            "id": f"event:{event_id}",
            "type": "MonetaryPolicyEvent" if event_id in ['rate_cut', 'rate_hike'] else "LaborMarketEvent",
            "name": event_display.get(event_id, event_id.replace('_', ' ').title()),
            "layer": 1,
            "attributes": {
                "event_class": event_id,
                "mention_count": summary['by_event'][event_id]['total_mentions']
            },
            "provenance": {
                "source": "News headlines",
                "created_at": timestamp
            }
        }
        kg["nodes"].append(event_node)
    
    # Layer 2: Mechanism Nodes
    for mech_id in sorted(relations['mechanisms']):
        mech_config = next((m for m in MECHANISM_KEYWORDS.values() if m['id'] == mech_id), None)
        if not mech_config:
            continue
        
        mech_data = summary['by_mechanism'][mech_id]
        dominant_polarity = max(mech_data['polarity'].items(), key=lambda x: x[1])[0] if mech_data['polarity'] else 'neutral'
        
        mech_node = {
            "id": mech_id,
            "type": "Mechanism",
            "name": mech_config['name'],
            "layer": 2,
            "attributes": {
                "mechanism_type": mech_config['type'],
                "dominant_polarity": dominant_polarity,
                "mention_count": mech_data['total_mentions']
            },
            "statistics": {
                "positive_mentions": mech_data['polarity'].get('positive', 0),
                "negative_mentions": mech_data['polarity'].get('negative', 0),
                "neutral_mentions": mech_data['polarity'].get('neutral', 0),
            },
            "provenance": {
                "source": "Derived from headlines",
                "created_at": timestamp
            }
        }
        kg["nodes"].append(mech_node)
    
    # Layer 3: Asset Nodes
    for asset_id in sorted(relations['assets']):
        asset_data = summary['by_asset'][asset_id]
        
        pos = asset_data['positive']
        neg = asset_data['negative']
        neu = asset_data['neutral']
        
        if pos > neg and pos > neu:
            dominant = "positive"
        elif neg > pos and neg > neu:
            dominant = "negative"
        else:
            dominant = "neutral"
        
        asset_node = {
            "id": f"asset:{asset_id}",
            "type": ASSET_TYPE_MAP.get(asset_id, "Asset"),
            "name": ASSET_DISPLAY_NAMES.get(asset_id, asset_id.replace('_', ' ').title()),
            "layer": 3,
            "aliases": ASSET_KEYWORDS.get(asset_id, []),
            "attributes": {
                "dominant_polarity": dominant,
                "relationship_count": asset_data['total']
            },
            "statistics": {
                "positive_mentions": pos,
                "negative_mentions": neg,
                "neutral_mentions": neu,
                "total_mentions": asset_data['total']
            },
            "provenance": {
                "source": "Derived from headlines",
                "created_at": timestamp
            }
        }
        kg["nodes"].append(asset_node)
    
    # Build Edges
    edge_id = 1
    
    # Event -> Mechanism and Event -> Asset edges
    event_edge_groups = defaultdict(lambda: {
        'target_type': None,
        'polarity': None,
        'evidence': [],
        'patterns': defaultdict(int)
    })
    
    for edge in relations['event_edges']:
        event, target, target_type, direction, title, pattern, date, url = edge
        key = (event, target, target_type)
        event_edge_groups[key]['target_type'] = target_type
        event_edge_groups[key]['polarity'] = direction
        event_edge_groups[key]['evidence'].append({
            'title': title,
            'date': str(date) if date and pd.notna(date) else None,
            'url': url if url and pd.notna(url) else None,
            'pattern': pattern
        })
        event_edge_groups[key]['patterns'][pattern] += 1
    
    for (event, target, target_type), data in event_edge_groups.items():
        evidence_count = len(data['evidence'])
        most_common_pattern = max(data['patterns'].items(), key=lambda x: x[1])[0]
        
        if target_type == 'mechanism':
            relation = 'TRIGGERS'
            target_id = target
        else:  # asset
            relation = 'POSITIVELY_IMPACTS' if data['polarity'] == 'positive' else \
                       'NEGATIVELY_IMPACTS' if data['polarity'] == 'negative' else 'INDIRECTLY_AFFECTS'
            target_id = f"asset:{target}"
        
        edge = {
            "id": f"edge:e{edge_id}",
            "type": "Causal",
            "source": f"event:{event}",
            "target": target_id,
            "relation": relation,
            "polarity": data['polarity'],
            "evidence_count": evidence_count,
            "primary_pattern": most_common_pattern,
            "pattern_distribution": dict(data['patterns']),
            "evidence": data['evidence'][:10],
            "last_updated": timestamp,
        }
        kg["edges"].append(edge)
        edge_id += 1
    
    # Mechanism -> Asset edges
    mech_edge_groups = defaultdict(lambda: {
        'polarity': None,
        'evidence': [],
        'patterns': defaultdict(int)
    })
    
    for edge in relations['mechanism_edges']:
        mechanism, asset, direction, title, pattern, date, url = edge
        key = (mechanism, asset)
        mech_edge_groups[key]['polarity'] = direction
        mech_edge_groups[key]['evidence'].append({
            'title': title,
            'date': str(date) if date and pd.notna(date) else None,
            'url': url if url and pd.notna(url) else None,
            'pattern': pattern
        })
        mech_edge_groups[key]['patterns'][pattern] += 1
    
    for (mechanism, asset), data in mech_edge_groups.items():
        evidence_count = len(data['evidence'])
        most_common_pattern = max(data['patterns'].items(), key=lambda x: x[1])[0]
        
        relation = 'POSITIVELY_IMPACTS' if data['polarity'] == 'positive' else \
                   'NEGATIVELY_IMPACTS' if data['polarity'] == 'negative' else 'INDIRECTLY_AFFECTS'
        
        edge = {
            "id": f"edge:e{edge_id}",
            "type": "Causal",
            "source": mechanism,
            "target": f"asset:{asset}",
            "relation": relation,
            "polarity": data['polarity'],
            "evidence_count": evidence_count,
            "primary_pattern": most_common_pattern,
            "pattern_distribution": dict(data['patterns']),
            "evidence": data['evidence'][:10],
            "last_updated": timestamp,
        }
        kg["edges"].append(edge)
        edge_id += 1
    
    # Update metadata
    kg["metadata"]["total_nodes"] = len(kg["nodes"])
    kg["metadata"]["total_edges"] = len(kg["edges"])
    
    return kg


def export_to_csv(relations: Dict, output_dir: str = 'output'):
    """
    Export relationships to CSV format.
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all edges
    all_edges = []
    
    for edge in relations['event_edges']:
        event, target, target_type, direction, title, pattern, date, url = edge
        all_edges.append({
            'source': event,
            'source_type': 'event',
            'target': target,
            'target_type': target_type,
            'relation': 'TRIGGERS' if target_type == 'mechanism' else 'IMPACTS',
            'polarity': direction,
            'title': title,
            'pattern': pattern,
            'date': date,
            'url': url
        })
    
    for edge in relations['mechanism_edges']:
        mechanism, asset, direction, title, pattern, date, url = edge
        all_edges.append({
            'source': mechanism,
            'source_type': 'mechanism',
            'target': asset,
            'target_type': 'asset',
            'relation': 'IMPACTS',
            'polarity': direction,
            'title': title,
            'pattern': pattern,
            'date': date,
            'url': url
        })
    
    df = pd.DataFrame(all_edges)
    output_path = os.path.join(output_dir, 'multi_event_causal_relationships.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n Relationships exported to: {output_path}")


def export_to_json(kg: Dict, output_dir: str = 'output'):
    """
    Export knowledge graph to JSON file in the output directory.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'multi_event_causal_kg.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    
    print(f"\n Knowledge graph exported to: {output_path}")
    print(f"  - {kg['metadata']['total_nodes']} nodes")
    print(f"  - {kg['metadata']['total_edges']} edges")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    Loads news headlines from multi_event.csv and constructs the knowledge graph.
    """
    import os
    
    print("\n" + "=" * 100)
    print("MULTI-EVENT CAUSAL KNOWLEDGE GRAPH EXTRACTION")
    print("=" * 100)
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV input file
    input_file = 'multi_event_mini.csv'
    
    if not os.path.exists(input_file):
        print(f"\nError: Input file '{input_file}' not found!")
        print(f"   Please ensure '{input_file}' is in the current directory.")
        return
    
    print(f"\nLoading headlines from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"CSV file loaded successfully")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Try to find article title column
        title_col = None
        for col in df.columns:
            if 'title' in col.lower() or 'headline' in col.lower():
                title_col = col
                break
        
        # If no title column found, use first column
        if not title_col:
            title_col = df.columns[0]
            print(f"\nNo 'title' or 'headline' column found.")
            print(f"  Using first column '{title_col}' as headlines.")
        else:
            print(f"Using column '{title_col}' for headlines")
        
        titles = df[title_col].tolist()
        print(f"Extracted {len(titles)} headlines from CSV")
        
    except Exception as e:
        print(f"\n Error loading CSV file: {e}")
        return
    
    print("\nExtracting causal relationships...")
    relations = extract_multi_event_relations(titles, df)
    
    print("\nGenerating summary statistics...")
    summary = summarize_relations(relations)
    
    print_extraction_results(relations, summary)
    
    # Export to CSV (in output folder)
    export_to_csv(relations, output_dir)
    
    # Build and export JSON knowledge graph (in output folder)
    print("\nBuilding JSON knowledge graph...")
    kg = build_multi_event_knowledge_graph(relations, summary)
    export_to_json(kg, output_dir)
    
    print("\n" + "=" * 100)
    print("EXTRACTION COMPLETE")
    print("=" * 100)
    print(f"\nInput file processed: {input_file}")
    print(f"Output files created in '{output_dir}/' folder:")
    print("  1. multi_event_causal_relationships.csv - Flat edge list")
    print("  2. multi_event_causal_kg.json - Full knowledge graph structure")


if __name__ == "__main__":
    main()
