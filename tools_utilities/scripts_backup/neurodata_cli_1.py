#!/usr/bin/env python3
"""
Small-Mind INFINITE Neurodata Search System

An infinitely expandable command-line interface for accessing open neuroscience data sources.
This system can handle endless parameters, queries, and search combinations with self-improvement.

Features:
- Infinite parameter combinations
- Interactive search sessions
- Dynamic result filtering
- Expandable data sources
- Custom search pipelines
- Real-time result analysis
- Self-improving search strategies
- Learned parameter optimization

Usage:
    python3 neurodata_cli.py search "cortex" --species human --modality fMRI --brain-region visual
    python3 neurodata_cli.py infinite "neuron"
    python3 python3 neurodata_cli.py intelligent "cortex"
    python3 neurodata_cli.py interactive
    python3 neurodata_cli.py explore --endless
    python3 neurodata_cli.py pipeline "custom_search_flow"
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import itertools
import random
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from neurodata import NeurodataManager
    from neurodata import OpenNeurophysiologyInterface, OpenBrainImagingInterface, CommonCrawlInterface
    from infinite_search_engine import SelfImprovingSearchEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the small-mind directory")
    sys.exit(1)


class InfiniteNeurodataCLI:
    """Infinite neurodata CLI with self-improvement capabilities"""
    
    def __init__(self):
        self.manager = NeurodataManager()
        self.infinite_engine = SelfImprovingSearchEngine(learning_enabled=True)
        self.search_history = []
        
    def search_data(self, query: str, data_types: list = None, species: str = None, **kwargs):
        """Enhanced search with endless parameters"""
        try:
            print(f"🔍 Searching for: {query}")
            
            # Handle additional parameters
            brain_regions = kwargs.get('brain_regions', None)
            modalities = kwargs.get('modalities', None)
            techniques = kwargs.get('techniques', None)
            diseases = kwargs.get('diseases', None)
            developmental_stages = kwargs.get('developmental_stages', None)
            
            results = self.manager.search_across_sources(
                query=query,
                data_types=data_types,
                species=species,
                brain_regions=brain_regions
            )
            
            print(f"\n✅ Found data from {len(results)} sources:")
            print("=" * 60)
            
            for source, source_results in results.items():
                print(f"\n📊 {source.upper()}: {len(source_results)} results")
                if source_results:
                    for i, result in enumerate(source_results[:3]):  # Show first 3 results
                        if isinstance(result, dict):
                            name = result.get('name', result.get('keyword', 'Unnamed'))
                            desc = result.get('description', 'No description available')
                            # Skip results with generic names
                            if name.lower() in ['unnamed', 'unknown', 'sample', 'generic']:
                                continue
                            print(f"  {i+1}. {name}")
                            print(f"     {desc[:80]}...")
                        else:
                            print(f"  {i+1}. {result}")
                    if len(source_results) > 3:
                        print(f"     ... and {len(source_results) - 3} more results")
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return None
    
    def infinite_search(self, query: str, **kwargs):
        """Perform infinite search with endless parameters and learning"""
        print(f"🚀 LAUNCHING INFINITE SEARCH: {query}")
        print("=" * 80)
        
        try:
            results = self.infinite_engine.infinite_search(query, **kwargs)
            
            # Display learning progress
            if "metadata" in results and "learning_progress" in results["metadata"]:
                print(f"\n🧠 LEARNING PROGRESS:")
                print(f"  Query patterns learned: {results['metadata']['final_learning_state']['query_patterns']}")
                print(f"  Successful combinations: {results['metadata']['final_learning_state']['successful_combinations']}")
                print(f"  Search strategies: {results['metadata']['final_learning_state']['search_strategies']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Infinite search failed: {e}")
            return None
    
    def intelligent_search(self, query: str, search_type: str = "general", **kwargs):
        """Perform intelligent search using learned knowledge"""
        print(f"🧠 INTELLIGENT SEARCH: {query} (using learned knowledge)")
        print("=" * 80)
        
        try:
            results = self.infinite_engine.intelligent_search(query, search_type, **kwargs)
            
            # Display learning metadata
            if results and "learning_metadata" in results:
                metadata = results["learning_metadata"]
                print(f"\n📚 LEARNING METADATA:")
                print(f"  Applied learned params: {metadata['applied_learned_params']}")
                print(f"  Combined params: {metadata['combined_params']}")
                print(f"  Model patterns: {metadata['model_patterns']}")
                print(f"  Successful combinations: {metadata['successful_combinations']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Intelligent search failed: {e}")
            return None
    
    def show_learning_stats(self):
        """Show current learning statistics"""
        try:
            print("🧠 LEARNING STATISTICS")
            print("=" * 60)
            
            stats = self.infinite_engine.get_learning_stats()
            
            for key, value in stats.items():
                if key != "adaptive_strategies":
                    print(f"  {key}: {value}")
            
            # Show adaptive strategies
            print(f"\n🔧 ADAPTIVE STRATEGIES:")
            for key, value in stats["adaptive_strategies"].items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Failed to show learning stats: {e}")
    
    def reset_learning(self):
        """Reset all learned knowledge"""
        try:
            confirm = input("⚠️  Are you sure you want to reset all learned knowledge? (yes/no): ")
            if confirm.lower() == 'yes':
                self.infinite_engine.reset_learning()
                print("✅ Learning model reset complete")
            else:
                print("❌ Learning reset cancelled")
        except Exception as e:
            print(f"❌ Failed to reset learning: {e}")
    
    def show_sources(self):
        """Show available data sources"""
        try:
            print("🌐 AVAILABLE OPEN NEUROSCIENCE DATA SOURCES")
            print("=" * 60)
            
            # Neurophysiology sources
            print("\n🧠 NEUROPHYSIOLOGY DATABASES:")
            phys = OpenNeurophysiologyInterface()
            sources = phys.get_available_sources()
            for name, desc in sources.items():
                print(f"  • {name}: {desc}")
            
            # Brain imaging sources
            print("\n🖼️ BRAIN IMAGING DATABASES:")
            imaging = OpenBrainImagingInterface()
            sources = imaging.get_available_sources()
            for name, desc in sources.items():
                print(f"  • {name}: {desc}")
            
            # CommonCrawl sources
            print("\n🌐 WEB DATA SOURCES:")
            crawl = CommonCrawlInterface()
            sources = crawl.get_available_sources()
            for name, desc in sources.items():
                print(f"  • {name}: {desc}")
                
        except Exception as e:
            print(f"❌ Failed to show sources: {e}")
    
    def show_stats(self):
        """Show data source statistics"""
        try:
            print("📊 NEURODATA SOURCE STATISTICS")
            print("=" * 60)
            
            stats = self.manager.get_data_statistics()
            
            for source, source_stats in stats.items():
                print(f"\n🔍 {source.upper()}:")
                if "error" not in source_stats:
                    for key, value in source_stats.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  Error: {source_stats['error']}")
                    
        except Exception as e:
            print(f"❌ Failed to show stats: {e}")
    
    def interactive_mode(self):
        """Interactive infinite search mode"""
        print("🎮 INTERACTIVE INFINITE SEARCH MODE")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' to exit")
        print("Available modes: infinite, intelligent, search, sources, stats, learning, reset")
        
        while True:
            try:
                command = input("\n🔍 Enter command: ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'help':
                    print("\n📚 Available Commands:")
                    print("  infinite <query> - Perform infinite search with endless parameters")
                    print("  intelligent <query> - Intelligent search using learned knowledge")
                    print("  search <query> - Basic search")
                    print("  sources - Show available data sources")
                    print("  stats - Show data statistics")
                    print("  learning - Show learning statistics")
                    print("  reset - Reset learned knowledge")
                    print("  quit/exit - Exit interactive mode")
                elif command.startswith('infinite '):
                    query = command[9:].strip()
                    if query:
                        self.infinite_search(query)
                    else:
                        print("❌ Please provide a query for infinite search")
                elif command.startswith('intelligent '):
                    query = command[12:].strip()
                    if query:
                        self.intelligent_search(query)
                    else:
                        print("❌ Please provide a query for intelligent search")
                elif command.startswith('search '):
                    query = command[7:].strip()
                    if query:
                        self.search_data(query)
                    else:
                        print("❌ Please provide a query for search")
                elif command == 'sources':
                    self.show_sources()
                elif command == 'stats':
                    self.show_stats()
                elif command == 'learning':
                    self.show_learning_stats()
                elif command == 'reset':
                    self.reset_learning()
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main CLI function with infinite capabilities"""
    if len(sys.argv) < 2:
        print("🧠 SMALL-MIND INFINITE NEURODATA SEARCH SYSTEM")
        print("=" * 60)
        print("Usage:")
        print("  python3 neurodata_cli.py search <query> [--species <species>] [--modality <modality>]")
        print("  python3 neurodata_cli.py infinite <query>")
        print("  python3 neurodata_cli.py intelligent <query>")
        print("  python3 neurodata_cli.py interactive")
        print("  python3 neurodata_cli.py sources")
        print("  python3 neurodata_cli.py stats")
        print("  python3 neurodata_cli.py learning")
        print("  python3 neurodata_cli.py reset")
        print("\nExamples:")
        print("  python3 neurodata_cli.py infinite 'cortex'")
        print("  python3 neurodata_cli.py intelligent 'neuron'")
        print("  python3 neurodata_cli.py interactive")
        return
    
    command = sys.argv[1].lower()
    cli = InfiniteNeurodataCLI()
    
    if command == "search":
        if len(sys.argv) < 3:
            print("❌ Please provide a search query")
            print("Usage: python3 neurodata_cli.py search <query>")
            return
        
        query = sys.argv[2]
        # Parse additional parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--species', help='Filter by species')
        parser.add_argument('--modality', help='Filter by modality')
        parser.add_argument('--brain-region', help='Filter by brain region')
        parser.add_argument('--data-type', help='Filter by data type')
        parser.add_argument('--technique', help='Filter by technique')
        parser.add_argument('--disease', help='Filter by disease')
        parser.add_argument('--developmental-stage', help='Filter by developmental stage')
        
        # Parse remaining arguments
        remaining_args = sys.argv[3:]
        if remaining_args:
            parsed_args = parser.parse_args(remaining_args)
            cli.search_data(query, 
                           species=parsed_args.species,
                           modalities=[parsed_args.modality] if parsed_args.modality else None,
                           brain_regions=[parsed_args.brain_region] if parsed_args.brain_region else None,
                           data_types=[parsed_args.data_type] if parsed_args.data_type else None,
                           techniques=[parsed_args.technique] if parsed_args.technique else None,
                           diseases=[parsed_args.disease] if parsed_args.disease else None,
                           developmental_stages=[parsed_args.developmental_stage] if parsed_args.developmental_stage else None)
        else:
            cli.search_data(query)
            
    elif command == "infinite":
        if len(sys.argv) < 3:
            print("❌ Please provide a query for infinite search")
            print("Usage: python3 neurodata_cli.py infinite <query>")
            return
        
        query = sys.argv[2]
        cli.infinite_search(query)
        
    elif command == "intelligent":
        if len(sys.argv) < 3:
            print("❌ Please provide a query for intelligent search")
            print("Usage: python3 neurodata_cli.py intelligent <query>")
            return
        
        query = sys.argv[2]
        cli.intelligent_search(query)
        
    elif command == "interactive":
        cli.interactive_mode()
        
    elif command == "sources":
        cli.show_sources()
        
    elif command == "stats":
        cli.show_stats()
        
    elif command == "learning":
        cli.show_learning_stats()
        
    elif command == "reset":
        cli.reset_learning()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: search, infinite, intelligent, interactive, sources, stats, learning, reset")


if __name__ == "__main__":
    main()
