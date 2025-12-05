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
    
    # Generic market proxies (Priority 1 additions)
    'treasury': ['treasury', 'treasuries', 't-bond', 't-bonds', 'govt bond', 'government bond'],
    'stock_market': ['stock market', 'equity market', 'equities', 'market'],
    'risk_appetite': ['risk appetite', 'risk-seeking', 'appetite for risk', 'appetite'],
    'safe_haven': ['safe haven', 'flight to quality', 'safe assets', 'flight to safety'],
    
    # NEW: Additional generic terms for better coverage
    'market_general': ['market', 'markets', 'trading', 'bourses', 'exchanges', 'wall st', 'main street'],
    'investors': ['investors', 'investor', 'trader', 'traders', 'fund', 'funds', 'asset managers'],
    'shares_general': ['shares', 'share', 'stocks', 'stock'],
    'companies': ['companies', 'company', 'firms', 'firm', 'corporate', 'corporates', 'business', 'businesses'],
    'tech_sector': ['tech', 'technology', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'facebook', 'tesla', 'nvidia', 'semiconductor', 'chip'],
    
    # Generic market proxies (Priority 1 additions)
    'treasury': ['treasury', 'treasuries', 't-bond', 't-bonds', 'govt bond', 'government bond'],
    'stock_market': ['stock market', 'equity market', 'equities', 'market'],
    'risk_appetite': ['risk appetite', 'risk-seeking', 'appetite for risk', 'appetite'],
    'safe_haven': ['safe haven', 'flight to quality', 'safe assets', 'flight to safety'],
    
    # NEW: Additional generic terms for better coverage
    'market_general': ['market', 'markets', 'trading', 'bourses', 'exchanges', 'wall st', 'main street'],
    'investors': ['investors', 'investor', 'trader', 'traders', 'fund', 'funds', 'asset managers'],
    'shares_general': ['shares', 'share', 'stocks', 'stock'],
    'companies': ['companies', 'company', 'firms', 'firm', 'corporate', 'corporates', 'business', 'businesses'],
    'tech_sector': ['tech', 'technology', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'facebook', 'tesla', 'nvidia', 'semiconductor', 'chip'],
    
    # General equity markets
    'stocks': ['stocks', 'shares', 'equity', 'wall street', 'wall st'],
    'reit': ['reit', 'reits', 'real estate'],
    'homebuilders': ['homebuilders', 'homebuilder', 'home builders'],
    'futures': ['futures', 'index futures', 'stock futures', 'equity futures', 'e-mini'],
    'etfs': ['etf', 'etfs', 'exchange traded fund', 'exchange-traded'],
    'indexes': ['indexes', 'indices', 'market indexes', 'market indices'],
    
    # Major indices
    'sp500': ['s&p 500', 's&p', 'spx', 'sp500'],
    'dow': ['dow', 'djia', 'dow jones'],
    'nasdaq': ['nasdaq'],
    'nikkei': ['nikkei'],
    'ftse': ['ftse'],
    'dax': ['dax', 'german dax'],
    'cac': ['cac', 'cac 40'],
    'ibex': ['ibex'],
    'asx': ['asx', 'australian shares'],
    
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
    'bonds': ['bonds','bonds', 'treasuries', 'debt', 'treasury', 'gilt', 'bund', 'bond market'],
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
    
    # Volatility and sentiment (NEW - Priority 1)
    'vix': ['vix', 'fear gauge', 'fear index', 'volatility index', 'cboe volatility', 'fear meter', "wall st's fear", "wall street's fear"],
    'sentiment': ['sentiment', 'investor sentiment', 'market sentiment', 'risk appetite', 'risk sentiment'],
    'confidence': ['confidence', 'investor confidence', 'business confidence'],
    'fear': ['fear', 'fear index', 'fear gauge', 'fear meter'],
    
    # Market reaction proxies (NEW - Priority 2)
    'equity_market': ['equity market', 'equity markets', 'stock market'],
    'bond_market': ['bond market', 'bond markets'],
    'mortgage_rates': ['mortgage rates', 'mortgage market'],
    'financial_conditions': ['financial conditions', 'credit conditions'],
    
    # NEW: Specific sector ETFs/generics
    'cyclical_stocks': ['cyclical stocks', 'cyclicals'],
    'defensive_stocks': ['defensive stocks', 'defensive sectors'],
    
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
    'dax': 'EquityIndex',
    'cac': 'EquityIndex',
    'ibex': 'EquityIndex',
    'asx': 'EquityIndex',
    'futures': 'EquityMarket',
    'etfs': 'EquityMarket',
    'indexes': 'EquityIndex',
    'homebuilders': 'EquitySector',
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
    'vix': 'VolatilityIndex',
    'sentiment': 'MarketSentiment',
    # NEW asset types
    'market_general': 'EquityMarket',
    'investors': 'MarketParticipant',
    'shares_general': 'EquityMarket',
    'companies': 'Corporate',
    'tech_sector': 'EquitySector',
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
    'dax': 'DAX (Germany)',
    'cac': 'CAC 40 (France)',
    'ibex': 'IBEX 35 (Spain)',
    'asx': 'ASX (Australia)',
    'futures': 'Index Futures',
    'homebuilders': 'Homebuilders',
    'etfs': 'Exchange-Traded Funds',
    'indexes': 'Market Indexes',
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
    'vix': 'VIX (Volatility Index)',
    'sentiment': 'Market Sentiment',
    # NEW asset display names
    'market_general': 'Financial Markets (General)',
    'investors': 'Market Investors/Traders',
    'shares_general': 'Equity Shares (General)',
    'companies': 'Corporate Sector',
    'tech_sector': 'Technology Sector',
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
        'crushed expectations', 'stronger than expected', 'vault', 'catapult',
        'skyrocket', 'leap'
    ],
    'positive': [
        'gain', 'rise', 'climb', 'advance', 'improve', 'boost', 'up', 'higher',
        'support', 'lift', 'strengthen', 'firm', 'extend gains', 'rebound',
        'better than', 'outperform', 'exceed', 'cheer', 'churred', 'edge up',
        'inches higher', 'ticks higher', 'nudge higher', 'firmer', 'firms',
        'add', 'adds', 'push higher', 'rising', 'buoy', 'buoys', 'lifted',
        'extend', 'extends', 'strengthen', 'strengthens', 'rose', 'gained',
        'advanced', 'rallied', 'climbed', 'improved', 'boosted'
    ],
    'weak_positive': [
        'edge up', 'inch up', 'recover', 'rebound', 'trim losses', 'pare losses',
        'stabilize', 'steady', 'hold', 'support', 'moderate gains', 'tick up',
        'nudge up', 'creep up', 'drift higher', 'hold gains'
    ],
    
    'strong_negative': [
        'plunge', 'crash', 'collapse', 'tumble', 'slammed', 'crushed', 'sink',
        'tanked', 'hammered', 'battered', 'selloff', 'massacre', 'bloodbath',
        'dismal', 'gloomy', 'grim', 'bleak', 'slump', 'crater', 'nosedive',
        'dive', 'plummet', 'crumble'
    ],
    'negative': [
        'fall', 'drop', 'decline', 'slip', 'slide', 'down', 'lower', 'weaken',
        'bruised', 'wounded', 'hit', 'off', 'dip', 'ease', 'soften', 'pressure',
        'bearish', 'miss', 'disappoint', 'worse than', 'underperform', 'sinks',
        'falls', 'drops', 'declines', 'slips', 'slides', 'weakens', 'eases',
        'dips', 'softens', 'shed', 'sheds', 'trim', 'trims', 'pare', 'pares',
        'give back', 'gives back', 'pull back', 'pulls back', 'lose', 'loses',
        'lost', 'erode', 'erodes', 'sag', 'sags', 'wane', 'wanes', 'falter',
        'falters', 'stumble', 'stumbles', 'buckle', 'buckles', 'slipped',
        'dropped', 'fell', 'declined', 'weakened', 'eased', 'dipped', 'sagged',
        'erased', 'erase', 'erases'
    ],
    'weak_negative': [
        'edge down', 'pare gains', 'retreat', 'modest decline', 'consolidate',
        'pullback', 'correction', 'cautious', 'wary', 'inch down', 'tick down',
        'nudge lower', 'drift lower', 'creep down'
    ],
    
    'neutral': [
        'flat', 'steady', 'stable', 'unchanged', 'mixed', 'choppy', 'volatile',
        'little changed', 'narrowly', 'range-bound', 'muted', 'subdued', 'hover',
        'hovers', 'linger', 'lingers', 'treading water', 'consolidate', 'consolidates',
        'hold steady', 'holds steady', 'remain flat', 'remains flat', 'stay flat', 'stays flat'
    ]
}


# ============================================================================
# ASSET-AWARE DIRECTION EXTRACTION
# ============================================================================

import re

def get_all_movement_positions(text: str) -> List[Tuple[int, int, str, str]]:
    """
    Find all movement indicators in text with their positions.
    
    Returns:
        List of (start, end, indicator, direction) tuples
    """
    text_lower = text.lower()
    movements = []
    
    # Order matters: check multi-word phrases first, then single words
    direction_indicators = [
        ('strong_positive', MOVEMENT_INDICATORS['strong_positive']),
        ('positive', MOVEMENT_INDICATORS['positive']),
        ('weak_positive', MOVEMENT_INDICATORS['weak_positive']),
        ('strong_negative', MOVEMENT_INDICATORS['strong_negative']),
        ('negative', MOVEMENT_INDICATORS['negative']),
        ('weak_negative', MOVEMENT_INDICATORS['weak_negative']),
        ('neutral', MOVEMENT_INDICATORS['neutral']),
    ]
    
    for direction, indicators in direction_indicators:
        # Sort by length (longest first) to match multi-word phrases first
        sorted_indicators = sorted(indicators, key=len, reverse=True)
        for indicator in sorted_indicators:
            # Use word boundary matching
            pattern = r'\b' + re.escape(indicator) + r'\b'
            for match in re.finditer(pattern, text_lower):
                # Normalize direction to simple positive/negative/neutral
                if 'positive' in direction:
                    simple_dir = 'positive'
                elif 'negative' in direction:
                    simple_dir = 'negative'
                else:
                    simple_dir = 'neutral'
                movements.append((match.start(), match.end(), indicator, simple_dir))
    
    # Remove overlapping matches (keep the first/longest one)
    movements = sorted(movements, key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlapping = []
    last_end = -1
    for start, end, indicator, direction in movements:
        if start >= last_end:
            non_overlapping.append((start, end, indicator, direction))
            last_end = end
    
    return non_overlapping


def get_asset_positions(text: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find all asset mentions in text with their positions.
    
    Returns:
        Dict mapping asset_id to list of (start, end) position tuples
    """
    text_lower = text.lower()
    asset_positions = {}
    
    for asset_id, keywords in ASSET_KEYWORDS.items():
        # Sort keywords by length (longest first) to match multi-word phrases first
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        for keyword in sorted_keywords:
            pattern = r'\b' + re.escape(keyword) + r"'?s?\b"  # Handle possessives
            for match in re.finditer(pattern, text_lower):
                if asset_id not in asset_positions:
                    asset_positions[asset_id] = []
                # Avoid duplicate positions for same asset
                pos = (match.start(), match.end())
                if pos not in asset_positions[asset_id]:
                    asset_positions[asset_id].append(pos)
    
    return asset_positions


def extract_asset_movement_pairs(text: str) -> Dict[str, str]:
    """
    Extract (asset, direction) pairs by analyzing text structure.
    Handles headlines with multiple assets moving in different directions.
    
    IMPORTANT: This function ONLY extracts DIRECTIONS per asset.
    Event/causality attribution is done at the HEADLINE level separately.
    This ensures that in "Dollar Surges While Gold Retreats on Strong Jobs Data":
    - Both Dollar and Gold will be linked to the employment event (headline-level)
    - But Dollar gets 'positive' direction, Gold gets 'negative' direction (clause-level)
    
    Strategy:
    1. First, get ALL assets mentioned anywhere in the headline
    2. Split on conjunctions to isolate clauses for DIRECTION extraction
    3. For each clause, find assets and their nearby movements
    4. Use proximity matching as fallback
    5. Ensure ALL detected assets have at least a neutral direction
    
    Returns:
        Dict mapping asset_id to direction ('positive', 'negative', 'neutral')
    """
    text_lower = text.lower()
    asset_movements = {}
    
    # First, get ALL assets mentioned anywhere in the headline
    # This ensures no asset is "orphaned" due to clause splitting
    all_assets = detect_assets(text_lower)
    
    # Strategy 1: Split on major conjunctions and punctuation for DIRECTION extraction
    # This handles: "Dollar Surges While Gold Retreats"
    split_patterns = [
        r'\s+while\s+',
        r'\s+as\s+',
        r'\s+even as\s+',
        r'\s+whereas\s+',
        r'\s+but\s+',
        r'\s+yet\s+',
        r'\s+meanwhile\s+',
        r'\s+however\s+',
        r'\s+though\s+',
        r'\s+although\s+',
        r';\s*',  # Semicolon separator
        r',\s+and\s+',  # Comma-and separator
    ]
    
    # Build combined split pattern
    combined_pattern = '|'.join(f'({p})' for p in split_patterns)
    
    # Split text into clauses
    clauses = re.split(combined_pattern, text_lower, flags=re.IGNORECASE)
    # Filter out None and separator matches
    clauses = [c.strip() for c in clauses if c and c.strip() and len(c.strip()) > 3]
    
    # If no splits found, treat entire text as one clause
    if not clauses:
        clauses = [text_lower]
    
    # Track which assets have been assigned directions
    assets_with_directions = set()
    
    # Process each clause for DIRECTION extraction only
    for clause in clauses:
        clause_assets = set()
        clause_direction = 'neutral'
        
        # Find assets in this clause
        for asset_id, keywords in ASSET_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r"'?s?\b", clause):
                    clause_assets.add(asset_id)
                    break  # Found this asset, move to next
        
        # Find direction in this clause
        clause_direction = infer_direction_from_clause(clause)
        
        # Assign direction to each asset found in this clause
        for asset in clause_assets:
            # Only override if we found a non-neutral direction or asset not yet assigned
            if asset not in asset_movements or clause_direction != 'neutral':
                asset_movements[asset] = clause_direction
                assets_with_directions.add(asset)
    
    # Strategy 2: Proximity-based refinement for assets not yet assigned direction
    # This catches assets that might have been missed by clause splitting
    remaining_assets = all_assets - assets_with_directions
    
    if remaining_assets or len(clauses) == 1:
        proximity_movements = extract_asset_movement_by_proximity(text)
        for asset, direction in proximity_movements.items():
            if asset not in asset_movements or asset_movements[asset] == 'neutral':
                asset_movements[asset] = direction
    
    # CRITICAL: Ensure ALL detected assets have at least a neutral direction
    # This prevents any asset from being "orphaned" due to clause splitting
    for asset in all_assets:
        if asset not in asset_movements:
            asset_movements[asset] = 'neutral'
    
    return asset_movements


def infer_direction_from_clause(clause: str) -> str:
    """
    Infer direction from a single clause (no conjunction interference).
    """
    clause_lower = clause.lower()
    
    # Check for strong negative first (more specific)
    for indicator in MOVEMENT_INDICATORS['strong_negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'negative'
    
    for indicator in MOVEMENT_INDICATORS['negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'negative'
    
    for indicator in MOVEMENT_INDICATORS['weak_negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'negative'
    
    # Check for strong positive
    for indicator in MOVEMENT_INDICATORS['strong_positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'positive'
    
    for indicator in MOVEMENT_INDICATORS['positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'positive'
    
    for indicator in MOVEMENT_INDICATORS['weak_positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'positive'
    
    # Check neutral
    for indicator in MOVEMENT_INDICATORS['neutral']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', clause_lower):
            return 'neutral'
    
    return 'neutral'


def extract_asset_movement_by_proximity(text: str) -> Dict[str, str]:
    """
    Match movements to assets using proximity/word distance.
    
    For each asset, find the closest movement indicator and assign that direction.
    Forward movements (after asset mention) are slightly preferred.
    
    Returns:
        Dict mapping asset_id to direction
    """
    text_lower = text.lower()
    asset_movements = {}
    
    # Get all asset positions
    asset_positions = get_asset_positions(text_lower)
    
    # Get all movement positions
    movement_positions = get_all_movement_positions(text_lower)
    
    if not movement_positions:
        return asset_movements
    
    # For each asset, find closest movement indicator
    for asset_id, positions in asset_positions.items():
        closest_direction = 'neutral'
        closest_distance = float('inf')
        
        for asset_start, asset_end in positions:
            asset_center = (asset_start + asset_end) / 2
            
            for move_start, move_end, indicator, direction in movement_positions:
                move_center = (move_start + move_end) / 2
                distance = abs(asset_center - move_center)
                
                # Slight preference for movement that comes after asset mention
                # (e.g., "Dollar surges" vs "surging Dollar")
                if move_center > asset_center:
                    distance *= 0.9  # 10% boost for forward movement
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_direction = direction
        
        if closest_distance < float('inf'):
            asset_movements[asset_id] = closest_direction
    
    return asset_movements


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
    # NEW: Opens door to pattern (conditional causality) - HIGH IMPACT
    {
        'name': 'opens_door_to',
        'pattern': r'(opens? (the )?door (to|for)|paves? (the )?way (to|for)|clears? path (to|for)|signals?|suggests?)\s+.*?(rate (hike|cut)|fed (hike|cut)|policy)',
        'requires_movement': False,
        'priority': 10
    },
    # NEW: Reveals/Shows pattern (information disclosure) - HIGH IMPACT
    {
        'name': 'reveals_shows',
        'pattern': r'(reveals?|shows?|indicates?|suggests?|signals?)\s+.*?(economy|growth|labor|employment)',
        'requires_movement': False,
        'priority': 9
    },
    # NEW: Generic market reaction - HIGH IMPACT
    {
        'name': 'generic_market_reaction',
        'pattern': r'(stock market|equities?|shares|market)\s+(reacts?|responds?|opens?|closes?)\s+(higher|lower|up|down)\s+.*?(jobs|employment|rate)',
        'requires_movement': True,
        'priority': 9
    },
    # NEW: Sentiment lift/dip pattern - HIGH IMPACT
    {
        'name': 'sentiment_lift_dip',
        'pattern': r'(sentiment|confidence|mood)\s+(lift|dip|surge|fall|rises?|drops?)\s+.*?(jobs|employment)',
        'requires_movement': False,
        'priority': 9
    },
    # NEW: Fear gauge inverse pattern - MEDIUM IMPACT
    {
        'name': 'fear_gauge_inverse',
        'pattern': r'(fear gauge|vix|volatility)\s+(dips?|falls?|rises?|spikes?)\s+.*?(jobs|employment|rate cut)',
        'requires_movement': True,
        'priority': 8
    },
    
    # CRITICAL FIX #1: Passive voice pattern (Headline #1: "shares lifted BY jobs report")
    {
        'name': 'passive_voice_by',
        'pattern': r'(shares?|stocks?|dollar|bond|gold|yield|market|equit|treasur|currencies?)\s+\w+\s+by\s+.*?(jobs?|employment|payroll|rate cut|rate hike|fed)',
        'requires_movement': True,
        'priority': 10
    },
    
    # CRITICAL FIX #2: Before/Ahead of temporal patterns (Headlines #2, #5)
    # These handle REVERSE temporal direction: [movement] BEFORE/AHEAD OF [event]
    {
        'name': 'movement_before_event',
        'pattern': r'(shares?|stocks?|dollar|bond|gold|yield|market|equit|treasur)\s+(rose?|fell|climbed|dropped|gained|lost|rallied|slumped|surged|plunged|advanced|declined|dropped|soar|jump|spike|drifts?|ticks?|edges?|inch)\s+(before|ahead of|in advance of|prior to)\s+.*(jobs?|employment|payroll|rate cut|rate hike|fed)',
        'priority': 10
    },
    
    # CRITICAL FIX #3: Expanded movement verb list (Headlines #9, #10)
    # Added: jump, soar, spike, drifts, ticks, edges, inches, leap, dive, wanes, ebbs, pares, trims
    {
        'name': 'asset_movement_on_event',
        'pattern': r'(shares?|stocks?|dollar|bond|gold|yield|market|equit|treasur|currencies?)\s+(rose?|fell|climbed|dropped|gained|lost|rallied|slumped|surged|plunged|advanced|declined|weakened|strengthened|soar|soars|soared|jump|jumps|jumped|spike|spikes|spiked|drifts?|drifting|ticks?|ticking|edges?|edging|inches?|inching|leap|leaps|leaped|dive|dives|dived|wanes?|waning|ebbs?|ebbing|pares?|paring|trims?|trimming|extends?|extending)\s+(on|after|following|as|amid|with)\s+.*(jobs?|employment|payroll|rate cut|rate hike|fed|labor|labour)',
        'priority': 10
    },
    {
        'name': 'asset_direction_ahead',
        'pattern': r'(stocks?|dollar|bond|gold|yield|market|shares?)\s+(higher|lower|up|down|rise|fall|gain|decline)\s+(ahead of|before|awaiting?|eyes? on|focus.* on)',
        'priority': 10
    },
    {
        'name': 'movement_despite_event',
        'pattern': r'(stocks?|dollar|bond|gold|yield|market|shares?)\s+(rose?|gained|climbed|advanced|rallied).*(despite|even as|notwithstanding|shrugs? off)',
        'priority': 9
    },
    {
        'name': 'event_quality_causes_movement',
        'pattern': r'(strong|weak|better|worse|upbeat|disappointing|solid|tepid|robust|dismal|blowout)\s+(jobs?|employment|payroll|labor|labour)\s+(report|data|reading|numbers?)',
        'requires_movement': True,  # Must have movement elsewhere in headline
        'priority': 10
    },
    {
        'name': 'simple_asset_event_cooccurrence',
        'pattern': r'(stocks?|dollar|bond|market|shares?).*(jobs?|employment|payroll|rate\s+cut|rate\s+hike)',
        'requires_movement': True,  # Needs movement word to be causal
        'priority': 7  # Lower priority, catches many cases
    },
    {
        'name': 'event_sends_asset_direction',
        'pattern': r'(jobs?|employment|payroll|rate|fed|data|report|reading).*(send|push|drive|lift|weigh|pull|drag|boost).*(stocks?|dollar|bond|gold|market|shares?).*(higher|lower|up|down)',
        'priority': 9
    },
    {
        'name': 'asset_reacts_to_event',
        'pattern': r'(stocks?|dollar|bond|market|shares?)\s+(react|respond|move|swing|sway|shift).*(jobs?|employment|payroll|rate|fed)',
        'priority': 8
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
        'pattern': r'(strong|weak|disappointing|better|worse|blowout|dismal|upbeat)\s+.*?(jobs report|employment).*?(send|lift|weigh|boost|hurt)\s+\w+\s+(higher|lower)',
        'priority': 8
    },
    {
        'name': 'movement_before',
        'pattern': r'(before|ahead of|awaiting?|anticipating?)\s+.*?(rate cut|jobs report|employment data)',
        'requires_movement': True,
        'priority': 8
    },
    # NEW: Eyes on pattern (forward-looking) - MEDIUM IMPACT
    {
        'name': 'eyes_on',
        'pattern': r'(eyes? on|focuses? on|watch(es|ing)?|awaits?)\s+.*?(jobs report|employment data|rate decision|fed)',
        'requires_movement': True,
        'priority': 8
    },
    # NEW: Lifts/Dips sentiment pattern - MEDIUM IMPACT
    {
        'name': 'lifts_dips_sentiment',
        'pattern': r'(lifts?|dips?|boosts?|weighs? on|dampens?|supports?|hurts?)\s+(sentiment|mood|confidence|optimism)',
        'requires_movement': False,
        'priority': 9
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
    
    # NEW PATTERNS - Additional flexible patterns for better coverage
    {
        'name': 'simple_before_after_jobs',
        'pattern': r'(before|after|ahead of|following|post).*?(jobs?|employment|payroll|labor|labour)',
        'requires_movement': True,
        'priority': 7
    },
    {
        'name': 'asset_event_simple_cooccur',
        'pattern': r'(stock|bond|market|dollar|gold|yield|crude|shares?|equit).*(jobs?|employment|payroll|unemployment|labor|labour)',
        'requires_movement': True,
        'priority': 6  # Lower priority, broad catch
    },
    {
        'name': 'event_asset_simple_cooccur',
        'pattern': r'(jobs?|employment|payroll|unemployment|labor|labour).*(stock|bond|market|dollar|gold|yield|crude|shares?|equit)',
        'requires_movement': True,
        'priority': 6  # Lower priority, broad catch
    },
    {
        'name': 'set_to_open_pattern',
        'pattern': r'(set to|seen|expected|poised|likely to)\s+(open|close|move|rise|fall)',
        'requires_movement': False,
        'priority': 7
    },
    {
        'name': 'keeps_on_track',
        'pattern': r'(keeps?|puts?|maintains?).*(on track|on course|on path)',
        'requires_movement': False,
        'priority': 7
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
    text_lower = text.lower()
    
    for mech_key, mech_config in MECHANISM_KEYWORDS.items():
        for pattern in mech_config['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.add(mech_config['id'])
                break  # Found this mechanism, move to next
    
    return detected


def detect_employment_strength(text: str) -> Optional[str]:
    """
    Detect whether employment data is characterized as strong/weak/mixed.
    Enhanced with Priority 3 qualifiers.
    
    Returns:
        'strong', 'weak', 'mixed', or None
    """
    text_lower = text.lower()
    
    strong_indicators = [
        r'strong.*job', r'robust.*job', r'blowout.*job', r'solid.*job',
        r'beat.*expect', r'exceed.*expect', r'better.*than.*expect',
        r'surprise.*job', r'stunning.*job', r'crush', r'smashing',
        r'upbeat.*job', r'upbeat.*employment', r'upbeat.*labor',
        r'upbeat.*payroll', r'upbeat.*nonfarm',
        r'tight.*labor', r'labor.*tight', r'market.*tighten',
        r'labor.*improv', r'employment.*improv',
        r'claims.*fall', r'claims.*drop', r'claims.*declin',
        r'payrolls.*beat', r'payrolls.*exceed',
        r'stronger.*than.*expect', r'larger.*than.*expect',
        r'jobs.*added', r'employment.*growth',
        r'labor.*resilient', r'labor.*robust',
        r'nonfarm.*rise', r'payroll.*rise',
        r'payroll.*beat', r'payrolls?.*beat', r'jobs?.*beat',
        r'robust.*hiring', r'strong.*hiring',
        r'labor.*strength', r'employment.*strength',
        r'healthy.*labor', r'healthy.*employment',
    ]
    
    weak_indicators = [
        r'weak.*job', r'disappointing.*job', r'miss.*expect',
        r'soft.*job', r'tepid.*job', r'dismal.*job',
        r'worse.*than.*expect', r'below.*expect',
        r'claims.*rise', r'claims.*jump', r'claims.*surge',
        r'unemployment.*rise', r'unemployment.*climb',
        r'labor.*soften', r'labor.*weaken', r'labor.*cool',
        r'payrolls.*miss', r'payrolls.*disappoint',
        r'weaker.*than.*expect', r'smaller.*than.*expect',
        r'job.*loss', r'jobs?.*lost', r'layoff',
        r'labor.*slack', r'employment.*weak',
        r'gloomy.*job', r'grim.*job', r'bleak.*job',
        r'slowing.*hiring', r'hiring.*slow',
    ]
    
    has_strong = False
    has_weak = False
    
    for pattern in strong_indicators:
        if re.search(pattern, text_lower):
            has_strong = True
            break
    
    for pattern in weak_indicators:
        if re.search(pattern, text_lower):
            has_weak = True
            break
    
    if has_strong and has_weak:
        return 'mixed'
    elif has_strong:
        return 'strong'
    elif has_weak:
        return 'weak'
    
    return None


# ============================================================================
# ASSET DETECTION
# ============================================================================

def detect_assets(text: str) -> Set[str]:
    """
    Detect all asset types mentioned in the text.
    Returns set of asset type identifiers.
    
    Able to handle:
    - Multi-word phrases better
    - Possessives (e.g., "dollar's")
    - Case variations
    """
    text_lower = text.lower()
    detected = set()
    
    for asset_id, keywords in ASSET_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundary matching with possessive handling
            pattern = r'\b' + re.escape(keyword) + r"'?s?\b"
            if re.search(pattern, text_lower):
                detected.add(asset_id)
                break  # Found this asset type, move to next
    
    return detected


def infer_direction_from_movement(text: str) -> str:
    """
    Infer the overall direction from movement indicators in text.
    This is the fallback for when asset-specific direction is not available.
    
    Priority order: strong_negative > negative > weak_negative > 
                    strong_positive > positive > weak_positive > neutral
    
    Returns:
        'positive', 'negative', or 'neutral'
    """
    text_lower = text.lower()
    
    # Check for strong negative first (most impactful)
    for indicator in MOVEMENT_INDICATORS['strong_negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'negative'
    
    for indicator in MOVEMENT_INDICATORS['negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'negative'
    
    for indicator in MOVEMENT_INDICATORS['weak_negative']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'negative'
    
    # Check for strong positive
    for indicator in MOVEMENT_INDICATORS['strong_positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'positive'
    
    for indicator in MOVEMENT_INDICATORS['positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'positive'
    
    for indicator in MOVEMENT_INDICATORS['weak_positive']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'positive'
    
    # Check neutral
    for indicator in MOVEMENT_INDICATORS['neutral']:
        if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
            return 'neutral'
    
    return 'neutral'


def infer_direction_with_context(text: str, event_type: str, 
                                  employment_strength: Optional[str] = None,
                                  mechanisms: Set[str] = None,
                                  asset_type: str = None,
                                  base_direction: str = None) -> str:
    """
    Infer direction considering event type and mechanisms.
    
    Args:
        text: The headline text
        event_type: Type of event detected
        employment_strength: 'strong', 'weak', 'mixed', or None
        mechanisms: Set of mechanism IDs detected
        asset_type: The specific asset type being analyzed
        base_direction: Pre-computed direction for this specific asset (from proximity matching)
    
    Employment heuristics:
    - Weak jobs + rate cut bets → dovish → USD down, bonds up, gold up, stocks mixed-positive
    - Strong jobs + hike worries → hawkish → USD up, bonds down, gold down, stocks mixed-negative
    
    Returns:
        Direction string: 'positive', 'negative', or 'neutral'
    """
    # Use provided base_direction, or infer from full text as fallback
    if base_direction is not None:
        direction = base_direction
    else:
        direction = infer_direction_from_movement(text)
    
    if mechanisms is None:
        mechanisms = set()
    
    # Apply context-aware adjustments based on asset type and mechanisms
    # VIX has inverse relationship - if market is positive, VIX should be negative
    if asset_type == 'vix':
        # Check if headline explicitly mentions VIX direction
        text_lower = text.lower()
        vix_explicit = False
        for indicator in MOVEMENT_INDICATORS['positive'] + MOVEMENT_INDICATORS['strong_positive']:
            if 'vix' in text_lower and indicator in text_lower:
                vix_explicit = True
                break
        for indicator in MOVEMENT_INDICATORS['negative'] + MOVEMENT_INDICATORS['strong_negative']:
            if 'vix' in text_lower and indicator in text_lower:
                vix_explicit = True
                break
        
        # Only invert if VIX direction wasn't explicitly stated
        if not vix_explicit:
            if direction == 'positive':
                direction = 'negative'
            elif direction == 'negative':
                direction = 'positive'
    
    # Apply employment strength heuristics
    if employment_strength == 'strong':
        # Strong jobs typically positive for USD, negative for rate-sensitive assets
        if asset_type == 'dollar':
            if direction == 'neutral':
                direction = 'positive'
        elif asset_type in ['bonds', 'yields', 'gold']:
            if direction == 'neutral':
                direction = 'negative'
    elif employment_strength == 'weak':
        # Weak jobs typically negative for USD, positive for rate-sensitive assets (rate cut hopes)
        if asset_type == 'dollar':
            if direction == 'neutral':
                direction = 'negative'
        elif asset_type in ['bonds', 'gold']:
            if direction == 'neutral':
                direction = 'positive'
    
    # Apply mechanism-based adjustments
    if 'mech:rate_cut_bets' in mechanisms or 'mech:dovish_repricing' in mechanisms:
        if asset_type == 'dollar' and direction == 'neutral':
            direction = 'negative'
        elif asset_type in ['stocks', 'gold', 'bonds'] and direction == 'neutral':
            direction = 'positive'
    
    if 'mech:rate_hike_bets' in mechanisms or 'mech:hawkish_repricing' in mechanisms:
        if asset_type == 'dollar' and direction == 'neutral':
            direction = 'positive'
        elif asset_type in ['stocks', 'bonds'] and direction == 'neutral':
            direction = 'negative'
    
    return direction


def match_causal_pattern(text: str, asset_type: str) -> Optional[Tuple[str, str]]:
    """
    Match the text against causal patterns and return the best match.
    
    Args:
        text: The headline text (lowercase)
        asset_type: The asset type being analyzed
    
    Returns:
        Tuple of (pattern_name, inferred_direction) or None if no match
    """
    text_lower = text.lower()
    best_match = None
    best_priority = -1
    
    # Check if there's any movement indicator in the text (for patterns that require it)
    has_movement = False
    for direction_list in MOVEMENT_INDICATORS.values():
        for indicator in direction_list:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
                has_movement = True
                break
        if has_movement:
            break
    
    for pattern_config in CAUSAL_PATTERNS:
        pattern_name = pattern_config['name']
        pattern = pattern_config['pattern']
        priority = pattern_config.get('priority', 5)
        requires_movement = pattern_config.get('requires_movement', False)
        
        # Skip patterns that require movement if none found
        if requires_movement and not has_movement:
            continue
        
        if re.search(pattern, text_lower, re.IGNORECASE):
            if priority > best_priority:
                best_priority = priority
                # Infer direction from the pattern match context
                direction = infer_direction_from_movement(text_lower)
                best_match = (pattern_name, direction)
    
    return best_match


def extract_multi_event_relations(titles: List[str], df = None) -> Dict:
    """
    Extract multi-event causal relationships from a list of headlines.
    Now uses asset-aware direction extraction to handle multiple assets
    moving in different directions within the same headline.
    
    IMPORTANT DESIGN PRINCIPLE:
    - Events are detected at HEADLINE level (shared across ALL assets in that headline)
    - Directions are detected at CLAUSE/ASSET level (asset-specific)
    - ALL assets in a headline share the SAME causal event
    
    This ensures that in "Dollar Surges While Gold Retreats on Strong Jobs Data":
    - Both Dollar and Gold are linked to the employment event (headline-level causality)
    - But Dollar gets 'positive' direction, Gold gets 'negative' direction (clause-level direction)
    
    Args:
        titles: List of headline strings
        df: Optional DataFrame with additional metadata (date, url columns)
    
    Returns:
        Dictionary containing:
        - events: Set of detected event types
        - mechanisms: Set of detected mechanism IDs
        - assets: Set of detected asset types
        - event_edges: List of (event, target, target_type, direction, title, pattern, date, url) tuples
        - mechanism_edges: List of (mechanism, asset, direction, title, pattern, date, url) tuples
    """
    relations = {
        'events': set(),
        'mechanisms': set(),
        'assets': set(),
        'event_edges': [],
        'mechanism_edges': [],
    }
    
    for idx, title in enumerate(titles):
        if not title or not isinstance(title, str):
            continue
            
        title_lower = title.lower()
        
        # Get metadata from DataFrame if available
        date = None
        url = None
        if df is not None and idx < len(df):
            if 'date' in df.columns:
                date = str(df.iloc[idx]['date']) if pd.notna(df.iloc[idx]['date']) else None
            if 'url' in df.columns:
                url = str(df.iloc[idx]['url']) if pd.notna(df.iloc[idx]['url']) else None
        
        # =====================================================================
        # Step 1: Detect events at HEADLINE level (applies to ALL assets)
        # =====================================================================
        event_types = detect_event_type(title_lower)
        if not any(event_types.values()):
            continue  # Skip headlines without relevant events
        
        # Determine primary event (this applies to ALL assets in the headline)
        primary_event = None
        if event_types['employment']:
            primary_event = 'employment'
        elif event_types['rate_cut']:
            primary_event = 'rate_cut'
        elif event_types['rate_hike']:
            primary_event = 'rate_hike'
        
        if not primary_event:
            continue
        
        relations['events'].add(primary_event)
        
        # =====================================================================
        # Step 2: Detect mechanisms at HEADLINE level (shared context)
        # =====================================================================
        mechanisms = detect_mechanisms(title_lower)
        relations['mechanisms'].update(mechanisms)
        
        # =====================================================================
        # Step 3: Detect ALL assets in the headline
        # =====================================================================
        assets = detect_assets(title_lower)
        if not assets:
            continue  # Skip if no assets detected
        
        relations['assets'].update(assets)
        
        # =====================================================================
        # Step 4: Detect employment strength at HEADLINE level
        # =====================================================================
        employment_strength = None
        if primary_event == 'employment':
            employment_strength = detect_employment_strength(title_lower)
        
        # =====================================================================
        # Step 5: Extract ASSET-SPECIFIC directions using clause/proximity matching
        # Direction is per-asset, but the causal event is shared across all assets
        # =====================================================================
        asset_movements = extract_asset_movement_pairs(title)
        
        # =====================================================================
        # Step 6: Create edges - ALL assets linked to the SAME headline-level event
        # but with their own ASSET-SPECIFIC directions
        # =====================================================================
        for asset in assets:
            # Get asset-specific direction from proximity matching
            base_direction = asset_movements.get(asset, 'neutral')
            
            # Apply context-aware adjustments (VIX inverse, employment strength heuristics, etc.)
            direction = infer_direction_with_context(
                title_lower, 
                primary_event,
                employment_strength=employment_strength,
                mechanisms=mechanisms,
                asset_type=asset,
                base_direction=base_direction
            )
            
            # Match causal pattern for this asset
            result = match_causal_pattern(title_lower, asset)
            pattern_name = result[0] if result else 'co_occurrence'
            
            if mechanisms:
                # Path A: Event -> Mechanism -> Asset (3-layer)
                for mechanism in mechanisms:
                    # Event -> Mechanism edge
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
                    
                    # Mechanism -> Asset edge
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
                # Path B: Event -> Asset (2-layer direct)
                relations['event_edges'].append((
                    primary_event,
                    asset,
                    'asset',
                    direction,
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