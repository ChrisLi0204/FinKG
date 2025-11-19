"""
Random Title Reader for LLM Prompts

Reads random titles from a text file and formats them as LLM prompts.
Useful for batch processing financial headlines through language models.
"""

import random
import argparse
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime


class RandomTitleReader:
    """
    Reads random titles from a file and formats them for LLM processing.
    """
    
    def __init__(self, input_file: str, seed: int = None):
        """
        Initialize the reader.
        
        Args:
            input_file: Path to text file with titles (one per line)
            seed: Random seed for reproducibility
        """
        self.input_file = Path(input_file)
        self.seed = seed
        self.titles = []
        
        if seed is not None:
            random.seed(seed)
        
        self._load_titles()
    
    def _load_titles(self) -> None:
        """Load all titles from file (CSV or text)."""
        print(f"Loading titles from {self.input_file}...")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"File not found: {self.input_file}")
        
        try:
            if self.input_file.suffix.lower() == '.csv':
                # Load from CSV file
                df = pd.read_csv(self.input_file)
                # Try common column names for titles
                title_col = None
                for col in ['Article_title', 'title', 'Title', 'headline', 'Headline']:
                    if col in df.columns:
                        title_col = col
                        break
                
                if title_col is None:
                    raise ValueError(f"Could not find title column. Available columns: {df.columns.tolist()}")
                
                self.titles = df[title_col].dropna().astype(str).tolist()
                self.titles = [t.strip() for t in self.titles if t.strip()]
            else:
                # Load from text file
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    self.titles = [line.strip() for line in f if line.strip()]
            
            print(f"✓ Loaded {len(self.titles)} total titles")
        
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            raise
    
    def get_random_titles(self, n: int = 1000) -> List[str]:
        """
        Get n random titles from the file.
        
        Args:
            n: Number of titles to return (default: 1000)
        
        Returns:
            List of randomly selected titles
        """
        if n > len(self.titles):
            print(f"⚠ Requested {n} titles but only {len(self.titles)} available")
            n = len(self.titles)
        
        selected_titles = random.sample(self.titles, n)
        print(f"✓ Selected {n} random titles")
        
        return selected_titles
    
    def format_as_llm_prompt(self, titles: List[str], system_prompt: str = None) -> str:
        """
        Format titles as an LLM prompt.
        
        Args:
            titles: List of titles
            system_prompt: Optional custom system prompt
        
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = """You are a financial data analyst specializing in event extraction and knowledge graph construction.

Analyze the following financial headlines and for each one:
1. Identify macroeconomic events (e.g., rate cuts, inflation data, employment reports)
2. Identify affected assets (e.g., stocks, bonds, currencies, commodities)
3. Identify the causal relationship (e.g., "rate cut causes stock increase")
4. Extract this as a structured triplet: [Event → Relation → Asset]

Provide output in JSON format with the following structure:
{
    "title": "original headline",
    "events": ["event1", "event2"],
    "assets": ["asset1", "asset2"],
    "relations": [
        {"cause": "event", "relation": "causes_increase", "effect": "asset"}
    ],
    "confidence": 0.0-1.0
}
"""
        
        prompt = f"{system_prompt}\n\n"
        prompt += "="*80 + "\n"
        prompt += "TITLES TO ANALYZE:\n"
        prompt += "="*80 + "\n\n"
        
        for i, title in enumerate(titles, 1):
            prompt += f"{i}. {title}\n"
        
        prompt += "\n" + "="*80 + "\n"
        prompt += "ANALYSIS:\n"
        prompt += "="*80 + "\n"
        
        return prompt
    
    def format_as_jsonl_prompt(self, titles: List[str]) -> str:
        """
        Format titles as JSONL for batch LLM processing.
        
        Args:
            titles: List of titles
        
        Returns:
            JSONL formatted string (one JSON per line)
        """
        import json
        
        lines = []
        for title in titles:
            obj = {
                "custom_id": f"title-{len(lines)+1}",
                "params": {
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are a financial data analyst. Extract causal relationships 
from financial headlines as JSON with: title, events, assets, relations."""
                        },
                        {
                            "role": "user",
                            "content": f"Analyze this headline: {title}"
                        }
                    ]
                }
            }
            lines.append(json.dumps(obj))
        
        return "\n".join(lines)
    
    def save_prompt(self, prompt: str, output_file: str = None) -> str:
        """
        Save formatted prompt to file.
        
        Args:
            prompt: Formatted prompt string
            output_file: Output file path (default: auto-generated)
        
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"llm_prompt_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
            print(f"✓ Saved prompt to {output_file} ({file_size_mb:.2f} MB)")
            
            return output_file
        
        except Exception as e:
            print(f"✗ Error saving file: {e}")
            raise
    
    def create_batch_prompts(self, titles: List[str], batch_size: int = 100) -> List[str]:
        """
        Split titles into multiple prompts for batch processing.
        
        Args:
            titles: List of all titles
            batch_size: Number of titles per batch
        
        Returns:
            List of formatted prompts
        """
        prompts = []
        
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size]
            prompt = self.format_as_llm_prompt(batch)
            prompts.append(prompt)
            print(f"  - Created batch {len(prompts)} with {len(batch)} titles")
        
        return prompts


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Random Title Reader for LLM Prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read 1000 random titles from CSV and create LLM prompt
  python randomtotle.py --input multi_event_mini.csv --count 1000
  
  # Create JSONL format for batch API
  python randomtotle.py --input multi_event_mini.csv --count 1000 --format jsonl
  
  # Create multiple batches (100 titles each)
  python randomtotle.py --input multi_event_mini.csv --count 1000 --batch 100
  
  # Use specific seed for reproducibility
  python randomtotle.py --input multi_event_mini.csv --count 1000 --seed 42
  
  # Preview first 20 titles
  python randomtotle.py --input multi_event_mini.csv --count 1000 --preview 20
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV or text file with titles'
    )
    
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=1000,
        help='Number of random titles to select (default: 1000)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'jsonl'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        help='Create multiple batches of specified size'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--preview',
        type=int,
        help='Preview first N titles without saving'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("RANDOM TITLE READER FOR LLM PROMPTS")
    print("="*80 + "\n")
    
    try:
        # Initialize reader
        reader = RandomTitleReader(args.input, seed=args.seed)
        
        # Get random titles
        titles = reader.get_random_titles(args.count)
        
        # Preview mode
        if args.preview:
            print(f"\n✓ Preview of first {args.preview} titles:\n")
            for i, title in enumerate(titles[:args.preview], 1):
                print(f"  {i}. {title}")
            print()
            return
        
        # Batch mode
        if args.batch:
            print(f"\nCreating batches of {args.batch} titles each...\n")
            batch_prompts = reader.create_batch_prompts(titles, batch_size=args.batch)
            
            for i, prompt in enumerate(batch_prompts, 1):
                output_file = f"llm_prompt_batch_{i}.txt"
                reader.save_prompt(prompt, output_file)
            
            print(f"\n✓ Created {len(batch_prompts)} batch prompts")
        
        # Single prompt mode
        else:
            if args.format == 'jsonl':
                print(f"\nFormatting as JSONL...\n")
                prompt = reader.format_as_jsonl_prompt(titles)
                output_file = args.output or f"llm_prompt_batch.jsonl"
            else:
                print(f"\nFormatting as text prompt...\n")
                prompt = reader.format_as_llm_prompt(titles)
                output_file = args.output
            
            reader.save_prompt(prompt, output_file)
        
        print("\n" + "="*80)
        print("✓ PROCESSING COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())