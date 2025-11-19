# Multi-Event Causal Knowledge Graph Extraction for Financial Markets

## Overview

This project extracts **causal relationships** between multiple economic events (US rate cuts, employment data) and financial assets from news headlines. It implements a sophisticated **5-layer entity architecture** with explicit mechanism/context layers to capture market sentiment, expectations, and transmission channels.

The program transforms flat text headlines into a structured knowledge graph that reveals:
- **What events occurred** (monetary policy, labor market)
- **Why markets reacted** (mechanisms, sentiment, expectations)
- **How assets moved** (directional impacts across FX, equities, commodities, bonds)
- **Evidence of causality** (pattern matching with headline evidence)

---

## Architecture Overview

The system is built on a **5-layer hierarchical architecture** that progressively abstracts from raw text to causal relationships:

```
Layer 0: Provenance
    ↓
Layer 1: Events (Monetary Policy, Labor Market)
    ↓
Layer 2: Mechanisms/Context (Expectations, Sentiment, Channels)
    ↓
Layer 3: Assets/Markets (FX, Equities, Commodities, Bonds)
    ↓
Layer 4: Outcomes/Metrics (Quantitative validation)
```

### Layer Descriptions

#### **Layer 0: Provenance**
- Source of information (news headlines, dates, URLs)
- Tracks evidence chain for each relationship
- Enables validation and traceability
- Stored as metadata in each edge: `date`, `url`, `source`

#### **Layer 1: Events**
- **Monetary Policy Events**: Rate cuts, rate hikes, tapering, policy shifts
- **Labor Market Events**: Employment reports, unemployment data, jobless claims
- Key attributes:
  - Event class (e.g., "rate_cut", "employment")
  - Mention count (how frequently mentioned)
  - Event category (broad classification)

**Example nodes:**
```json
{
  "id": "event:rate_cut",
  "type": "MonetaryPolicyEvent",
  "name": "Rate Cut",
  "attributes": {
    "event_class": "rate_cut",
    "mention_count": 543
  }
}
```

#### **Layer 2: Mechanisms/Context** (The Innovation Layer)
This is the **critical layer** that explains *how* and *why* events impact markets. Instead of direct event-to-asset links, mechanisms capture:

**Mechanism Types:**

1. **Expectation_Timing**
   - `After Jobs Report` - Markets react to report confirmation
   - `Ahead of Jobs Report` - Anticipatory positioning before data release
   - Captures the temporal dimension of market expectations

2. **Policy_Expectation**
   - `Rate Cut Expectations (Positive)` - Market sees cut as likely and beneficial
   - `Rate Cut Concerns (Negative)` - Fears of premature cuts
   - `Rate Hike Expectations` - Markets price in tightening

3. **Policy_Repricing**
   - `Hawkish Policy Repricing` - Market reprices inflation/tightening expectations higher
   - `Dovish Policy Repricing` - Market reprices easing expectations higher
   - Captures sentiment shifts about central bank direction

4. **Labor_State**
   - `Tight Labor Market` - Strong hiring, low unemployment
   - `Weak Labor Market` - Poor hiring, high unemployment
   - `Low/High Unemployment Rate` - Specific labor state indicators

5. **Macro_Channel**
   - `Wage Pressure/Growth` - Labor market tightness driving wage inflation
   - Captures structural economic conditions

**Why mechanisms matter:**
- Direct link: "Employment → Stocks" (vague)
- With mechanism: "Employment → [Tight Labor Market] → [Hawkish Repricing] → Stocks DOWN" (precise causality)

**Example mechanism node:**
```json
{
  "id": "mech:after_jobs_report",
  "type": "Mechanism",
  "name": "After Jobs Report",
  "attributes": {
    "mechanism_type": "Expectation_Timing",
    "dominant_polarity": "negative",
    "mention_count": 1420
  },
  "statistics": {
    "positive_mentions": 565,
    "negative_mentions": 650,
    "neutral_mentions": 205
  }
}
```

#### **Layer 3: Assets/Markets**
Financial instruments that react to events through mechanisms:

**Asset Categories:**
- **Currencies**: USD, EUR, JPY, GBP, BRL, etc.
- **Equity Indices**: S&P 500, Dow Jones, Nasdaq, Nikkei, FTSE
- **Regional Markets**: Asian stocks, European stocks, Brazilian stocks
- **Fixed Income**: Bonds, Treasury yields (2Y, 10Y, 30Y)
- **Commodities**: Gold, Silver, Crude Oil, Copper
- **Sectors**: Tech, Financials, Energy, Retail, Healthcare
- **Other**: Emerging Markets, Real Estate (REITs)

**Asset attributes:**
- Dominant polarity (positive/negative/neutral)
- Relationship count (how many causal paths)
- Statistics: positive, negative, neutral mention counts

#### **Layer 4: Outcomes** (Optional)
- Quantitative metrics (price changes, returns)
- Economic indicators (GDP, inflation)
- Used for validation against actual market data

---

## The Extraction Logic: How Events Become Relationships

### Step 1: Event Detection

The program scans headlines for **event keywords**:

```python
RATE_CUT_KEYWORDS = [
    'rate cut', 'rate cuts', 'rate cut bets', 'fed cut', 'rate reduction', ...
]

EMPLOYMENT_KEYWORDS = [
    'nonfarm payrolls', 'payroll', 'jobs report', 'employment data',
    'unemployment rate', 'jobless claims', ...
]
```

**Example:**
- Headline: "Fed signals rate cut as employment growth slows"
- Detected: `event:rate_cut` + `event:employment`

### Step 2: Mechanism Detection

The program matches **regex patterns** to identify mechanisms present in the headline:

```python
MECHANISM_KEYWORDS = {
    'after_jobs': {
        'patterns': [
            r'after.*?(jobs report|employment data)',
            r'following.*?(jobs report|employment data)',
            r'post-?(jobs report|employment)',
        ]
    },
    'rate_cut_bets': {
        'patterns': [
            r'rate cut (hopes?|bets?|speculation)',
            r'(cut.*?chances?|chances?.*?cut)',
        ]
    },
    # ... more mechanisms
}
```

**Example:**
- Headline: "Stocks rally after jobs report beats expectations"
- Detected: `mech:after_jobs_report` + positive sentiment

### Step 3: Asset Detection

The program identifies all mentioned financial assets via keyword matching:

```python
ASSET_KEYWORDS = {
    'dollar': ['dollar', 'usd', 'greenback', 'dxy', 'us dollar'],
    'stocks': ['stocks', 'shares', 'equities', 'wall street'],
    'gold': ['gold', 'bullion', 'xau'],
    # ... 40+ asset types
}
```

**Example:**
- Headline: "Dollar falls as rate cut bets strengthen"
- Detected: `asset:dollar`, `asset:stocks` (implied), `mech:rate_cut_bets`

### Step 4: Direction Inference

The program determines if relationships are **positive, negative, or neutral** using movement indicators:

```python
MOVEMENT_INDICATORS = {
    'strong_positive': ['surge', 'soar', 'rally', 'jump', 'spike', ...],
    'positive': ['gain', 'rise', 'climb', 'advance', ...],
    'strong_negative': ['plunge', 'crash', 'collapse', 'tumble', ...],
    'negative': ['fall', 'drop', 'decline', 'slip', ...],
    'neutral': ['flat', 'steady', 'mixed', 'choppy', ...],
}
```

**Context-aware direction inference:**
- Weak jobs + dovish mechanism → negative for USD (because stimulus likely)
- Strong jobs + hawkish mechanism → positive for USD (because rate hikes likely)

**Example:**
- Headline: "Weak jobs report sends dollar lower"
- Direction: `negative` for dollar

### Step 5: Causal Pattern Matching

The program prioritizes causal patterns to distinguish correlation from causation:

```python
CAUSAL_PATTERNS = [
    {
        'name': 'explicit_on_after',
        'pattern': r'(on|after|following)\s+.*?(rate cut|jobs report)',
        'priority': 10  # Highest priority
    },
    {
        'name': 'explicit_due_to',
        'pattern': r'(due to|thanks to|because of)\s+.*?(rate cut)',
        'priority': 10
    },
    {
        'name': 'event_causes_asset',
        'pattern': r'(rate cut|jobs report)\s+(boosts?|sends?|drives?|pushes?)',
        'priority': 10
    },
    # ... more patterns with lower priorities
]
```

**Priority system:** Explicit causal language (priority 10) > Expectation patterns (priority 8) > General context (priority 6)

### Step 6: Multi-Path Relationship Building

The program creates two types of causal paths:

#### **Path A: Event → Mechanism → Asset**
```
Employment Report → [After Jobs Report] → Stocks
     ↓                  ↓                    ↓
  Positive       Expectation_Timing    Positive Impact
```

This path captures: "Strong employment report (released) confirmed market expectations, stocks rally"

#### **Path B: Event → Asset (Direct)**
```
Rate Cut → Bonds
   ↓        ↓
Positive  Positive Impact
```

This path captures: "Rate cut directly supports bond prices"

---

## Example: Complete Extraction Flow

### Important: Single-Pass Extraction

**All edges for ONE headline are created in a SINGLE pass, NOT iteratively:**
- One headline is processed ONCE through the entire pipeline
- All events, mechanisms, and assets are detected simultaneously
- All applicable edges are created at the same time
- No re-processing or multiple iterations

---

### Input Headline:
```
"Wall Street jumps on strong jobs report as rate cut bets intensify"
```

### Extraction Process (Single Pass):

**Step 1: Simultaneous Detection (All at Once)**

```python
# Processing ONE headline - all detection happens in parallel

headline = "Wall Street jumps on strong jobs report as rate cut bets intensify"

# PARALLEL DETECTION (happens together, not sequentially)
events_detected = ['employment', 'rate_cut']           # All events found
mechanisms_detected = ['unemployment_low', 'rate_cut_bets']  # All mechanisms found
assets_detected = ['stocks']                           # All assets found
direction_inferred = 'positive'                        # Direction determined once

# No re-processing of the same headline for different edge types!
# ✗ NOT: Process for event→mechanism, then process again for mechanism→asset
# ✓ YES: Detect all, create all edges in one go
```

**Step 2: Event → Mechanism Edges Created**

```python
# From the detected components, create Event→Mechanism edges
# These use Pattern Priority 10 (explicit causal language)

for event in events_detected:  # ['employment', 'rate_cut']
    for mechanism in mechanisms_detected:  # ['unemployment_low', 'rate_cut_bets']
        # Create edge: event → mechanism
        # This captures: "How did the event trigger this market condition?"
```

**Step 3: Mechanism → Asset Edges Created**

```python
# From the same detected components, create Mechanism→Asset edges
# These use Pattern Priority 6-8 (context-based inference)

for mechanism in mechanisms_detected:  # ['unemployment_low', 'rate_cut_bets']
    for asset in assets_detected:      # ['stocks']
        # Create edge: mechanism → asset
        # This captures: "How does this market condition impact the asset?"
```

**Step 4: Result - All Edges Created Together**

```python
# SINGLE HEADLINE produces FOUR edges simultaneously:

edges_created_this_pass = [
    Edge(event:employment → mech:unemployment_low),      # Event triggers condition
    Edge(event:rate_cut → mech:rate_cut_bets),          # Event triggers expectation
    Edge(mech:unemployment_low → asset:stocks),          # Condition impacts asset
    Edge(mech:rate_cut_bets → asset:stocks),            # Expectation impacts asset
]

# All 4 edges added to knowledge graph in ONE ITERATION
# Headline never processed again
```

### Detailed Output Edges:

```json
[
  {
    "id": "edge:e1",
    "type": "Causal",
    "source": "event:employment",
    "target": "mech:unemployment_low",
    "relation": "TRIGGERS",
    "polarity": "positive",
    "evidence_count": 1,
    "primary_pattern": "explicit_on_after",
    "evidence": [
      {
        "title": "Wall Street jumps on strong jobs report as rate cut bets intensify",
        "date": "2025-11-19",
        "pattern": "explicit_on_after"
      }
    ]
  },
  {
    "id": "edge:e2",
    "type": "Causal",
    "source": "event:rate_cut",
    "target": "mech:rate_cut_bets",
    "relation": "TRIGGERS",
    "polarity": "positive",
    "evidence_count": 1,
    "primary_pattern": "event_causes_asset"
  },
  {
    "id": "edge:e3",
    "type": "Causal",
    "source": "mech:unemployment_low",
    "target": "asset:stocks",
    "relation": "POSITIVELY_IMPACTS",
    "polarity": "positive",
    "evidence_count": 1,
    "primary_pattern": "general_context"
  },
  {
    "id": "edge:e4",
    "type": "Causal",
    "source": "mech:rate_cut_bets",
    "target": "asset:stocks",
    "relation": "POSITIVELY_IMPACTS",
    "polarity": "positive",
    "evidence_count": 1,
    "primary_pattern": "general_context"
  }
]
```

---

### Complete Path A Visualization

```
ONE HEADLINE INPUT
        ↓
   SINGLE PASS PROCESSING
        ↓
    ┌───────────────────────────────────────┐
    │ Step 1: DETECT ALL (Simultaneous)     │
    │ • Events: [employment, rate_cut]      │
    │ • Mechanisms: [unemploy_low, cut_bets]│
    │ • Assets: [stocks]                    │
    │ • Direction: positive                 │
    └───────────────────────────────────────┘
        ↓
    ┌───────────────────────────────────────┐
    │ Step 2: CREATE EVENT→MECH EDGES       │
    │ • employment → unemploy_low           │
    │ • rate_cut → cut_bets                 │
    └───────────────────────────────────────┘
        ↓
    ┌───────────────────────────────────────┐
    │ Step 3: CREATE MECH→ASSET EDGES       │
    │ • unemploy_low → stocks               │
    │ • cut_bets → stocks                   │
    └───────────────────────────────────────┘
        ↓
    FOUR EDGES OUTPUT (all from one headline)
```

---

### How Aggregation Works Across Multiple Headlines

When the same relationships appear in **different headlines**, they aggregate:

**Headline 1:**
```
"Strong jobs push stocks higher as rate cut bets grow"
→ Creates edge: mech:rate_cut_bets → asset:stocks (positive)
  evidence_count = 1
```

**Headline 2:**
```
"Rate cut expectations boost equities amid solid employment"
→ Creates same edge: mech:rate_cut_bets → asset:stocks (positive)
  This INCREMENTS the existing edge
  evidence_count now = 2
```

**Headline 3:**
```
"Jobs report fuels rate cut bets, stocks rally"
→ Creates same edge again: mech:rate_cut_bets → asset:stocks (positive)
  evidence_count now = 3
```

**Final Aggregated Edge in Knowledge Graph:**

```json
{
  "source": "mech:rate_cut_bets",
  "target": "asset:stocks",
  "relation": "POSITIVELY_IMPACTS",
  "polarity": "positive",
  "evidence_count": 3,
  "primary_pattern": "explicit_on_after",
  "pattern_distribution": {
    "explicit_on_after": 1,
    "event_causes_asset": 1,
    "general_context": 1
  }
}
```

**Key Point:** The relationship appears **3 times in 3 different headlines**, but it's stored **once** in the KG with `evidence_count: 3`

---

### Example 2: Complex Headline with Multiple Events and Mechanisms

```
"Employment surge and Fed rate cut signals drive dollar lower, 
boost equities and commodities amid global risk-on sentiment"
```

**Single Pass Processing:**

```
DETECTED COMPONENTS:
├── Events: [employment, rate_cut]
├── Mechanisms: [unemployment_low, rate_cut_bets, tight_labor_market]
├── Assets: [dollar, stocks, commodities, global_equities]
└── Direction: positive (for stocks/commodities), negative (for dollar)

EDGES CREATED IMMEDIATELY (NOT iteratively):

Event → Mechanism (Event Triggers):
├─ employment → unemployment_low
├─ employment → tight_labor_market
├─ rate_cut → rate_cut_bets
└─ rate_cut → rate_cut_bets (duplicate, will be merged)

Mechanism → Asset (Mechanism Impacts):
├─ unemployment_low → stocks
├─ unemployment_low → commodities
├─ rate_cut_bets → stocks
├─ rate_cut_bets → commodities
├─ rate_cut_bets → dollar (negative)
├─ tight_labor_market → stocks
└─ tight_labor_market → commodities

TOTAL: 3 event→mech edges + 7 mech→asset edges = 10 edges from ONE headline
```

**All 10 edges created in ONE PASS of the headline**

---

### Example 3: Simple Headline

```
"Fed cuts rates as stocks surge"
```

**Single Pass:**

```
DETECTED:
├── Events: [rate_cut]
├── Mechanisms: [rate_cut_bets]  (implied from context)
├── Assets: [stocks]
└── Direction: positive

EDGES CREATED (ONE PASS):
├─ rate_cut → rate_cut_bets        (1 edge)
└─ rate_cut_bets → stocks          (1 edge)
                                   ─────────
                                   2 edges total
```

---

### Processing Flow in Code

```python
def extract_multi_event_relations(titles: List[str]) -> Dict:
    """Extract from all headlines"""
    relations = {
        'event_edges': [],      # Event → Mechanism/Asset
        'mechanism_edges': [],  # Mechanism → Asset
    }
    
    for headline in titles:  # Process ONE headline at a time
        
        # SINGLE PASS: All detection happens here
        events = detect_event_type(headline)              # Detect ALL events
        mechanisms = detect_mechanisms(headline)          # Detect ALL mechanisms
        assets = detect_assets(headline)                  # Detect ALL assets
        direction = infer_direction_from_movement(headline)  # Determine direction ONCE
        
        # CREATE ALL EVENT→MECHANISM EDGES (not iteratively)
        for event in events:
            for mechanism in mechanisms:
                relations['event_edges'].append((
                    event, mechanism, 'mechanism', 
                    direction, headline, pattern
                ))
        
        # CREATE ALL MECHANISM→ASSET EDGES (not iteratively)
        for mechanism in mechanisms:
            for asset in assets:
                relations['mechanism_edges'].append((
                    mechanism, asset, direction, 
                    headline, pattern
                ))
        
        # Move to NEXT headline - this one is done
    
    return relations  # All edges from all headlines
```

**Critical Points:**
1. ✓ ONE headline → ONE pass through extraction
2. ✓ ALL events, mechanisms, assets detected TOGETHER
3. ✓ ALL applicable edges created IMMEDIATELY
4. ✗ NOT re-processing the same headline multiple times
5. ✗ NOT iterating through different edge types for the same headline

---

## Key Features & Implementation Details

### 1. **Multi-Event Support**
The program handles **multiple simultaneous events** in a single headline:
- Employment + Rate Cut expectations in one headline
- Creates both paths independently
- Aggregates evidence across paths
- Captures complex market narratives

### 2. **Mechanism Layer (The Innovation)**
Instead of naive direct links, mechanisms provide **semantic meaning**:
- Why did the market react? (Expectation, repricing, labor state)
- When did it react? (Before/after data, during policy meetings)
- What's the transmission channel? (Policy expectations, wage pressure)

### 3. **Evidence Aggregation**
Multiple mentions of same relationship:
```json
{
  "source": "mech:rate_cut_bets",
  "target": "asset:dollar",
  "evidence_count": 19,
  "pattern_distribution": {
    "explicit_on_after": 8,
    "general_context": 6,
    "event_causes_asset": 5
  }
}
```

Shows: 19 headlines mention this relationship, with 8 using explicit timing language, etc.

### 4. **Polarity Tracking**
For each mechanism, the program tracks:
- Positive mentions: 79
- Negative mentions: 35
- Neutral mentions: 20
- Dominant polarity: Positive

This reveals market **sentiment bias** toward rate cut expectations.

### 5. **Asset Type Classification**
Assets automatically categorized:
- Commodity, Currency, EquityIndex, EquitySector, RegionalEquity, FixedIncome, etc.
- Enables targeted analysis by asset class
- Reveals cross-asset correlations

---

## Data Flow & Processing

```
multi_event.csv (Input Headlines)
        ↓
[Headline Parser] - Extract title, date, URL
        ↓
[Event Detector] - Identify economic events
        ↓
[Mechanism Matcher] - Extract sentiment & context
        ↓
[Asset Recognizer] - Identify financial instruments
        ↓
[Direction Inference] - Determine positive/negative/neutral
        ↓
[Causal Pattern Matcher] - Weight evidence by pattern type
        ↓
[Relationship Builder] - Create event→mechanism→asset chains
        ↓
[Aggregator] - Combine evidence, count frequencies
        ↓
[JSON KG Builder] - Structure 5-layer knowledge graph
        ↓
multi_event_causal_kg.json (Output Knowledge Graph)
multi_event_causal_relationships.csv (Flat edge list)
```

---

## How to Run

### Prerequisites:
```bash
pip install pandas
```

### Execution:
```bash
python multi_event_kg_1.py
```

### Expected Output:
```
=========================================
MULTI-EVENT CAUSAL KNOWLEDGE GRAPH EXTRACTION
=========================================

Loading headlines from multi_event.csv...
✓ CSV file loaded successfully
  - Rows: 4861
  - Columns: ['Date', 'Article_title', 'Publisher', ...]

Extracting causal relationships...
Generating summary statistics...

=========================================
EVENT SUMMARY
=========================================

EMPLOYMENT
  Total mentions: 2802
  Top mechanisms: {...}
  Top assets: {...}

RATE_CUT
  Total mentions: 543
  Top mechanisms: {...}

=========================================
ASSET SUMMARY
=========================================

Asset                     Pos      Neg      Neu    Total    Dominant
US Dollar                 132      157       67      356      Negative
...

✓ Relationships exported to: output/multi_event_causal_relationships.csv
✓ Knowledge graph exported to: output/multi_event_causal_kg.json
  - 60 nodes
  - 251 edges
```

---

## Understanding the Results

### KG Statistics:
- **60 total nodes**: 3 events + 12 mechanisms + 45 assets
- **251 edges**: Causal relationships captured
- **Evidence-based**: Each edge backed by multiple headlines

### Key Insights from Example Output:

**Mechanism Dominance:**
```
After Jobs Report: 1420 mentions (650 negative, 565 positive)
→ Indicates: Jobs report releases trigger immediate market repricing
→ Dominated by negative sentiment: Markets often disappointed by results
```

**Asset Sensitivity:**
```
US Dollar: 356 mentions, 157 negative, 132 positive
→ Indicates: Dollar is sensitive to rate expectations
→ More often weakens on employment/rate cut news
```

**Rate Cut Expectations:**
```
Rate Cut Bets: 134 mentions, 79 positive, 35 negative
→ Indicates: Market generally bullish on rate cut prospects
→ Strongly supportive of risk assets
```


## Technical Implementation Notes

### Regex Pattern Priorities:
1. **Explicit causation** (10): Direct event→asset language
2. **Expectation + movement** (8): "As market expects... assets move"
3. **Temporal + event** (7): "Before/after event... movement"
4. **General context** (6): Loose associations

### Keyword Coverage:
- 50+ rate-related keywords
- 40+ employment keywords
- 60+ asset keywords
- 10+ mechanisms with multi-pattern detection

### Aggregation Strategy:
- **Evidence counting**: Sum headlines supporting each edge
- **Polarity aggregation**: Dominant sentiment across headlines
- **Pattern distribution**: Which linguistic patterns appear most

---

## Interpretation Guidelines

### Reading the Knowledge Graph:

**Strong relationships** (high evidence count):
- 100+ mentions → Robust, highly-discussed channel
- 50-100 mentions → Well-established relationship
- 10-50 mentions → Consistent but less prominent
- <10 mentions → Rare or contextual

**Polarity distribution** (split between positive/negative):
- 80/20 split → Strong consensus on direction
- 60/40 split → Consistent trend with exceptions
- 50/50 split → Contested/ambiguous relationship

**Pattern distribution**:
- Dominated by explicit patterns → Journalists clearly state causation
- Many general_context patterns → Relationship is implicit/inferred



## Contact & Questions

For questions about the extraction methodology, mechanism design, or knowledge graph interpretation, refer to the inline code comments in `multi_event_kg_1.py`.

The program is designed for **extensibility** — keyword lists, pattern priorities, and mechanism definitions can be easily modified for domain adaptation.