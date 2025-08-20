#!/usr/bin/env python3
"""
Startup script for the Universal Domain Database
"""

import sys
import os
from pathlib import Path

# Add the database directory to Python path
database_path = Path(__file__).parent
sys.path.insert(0, str(database_path))

from learning_engine.self_learning_system import SelfLearningSystem
from scrapers.internet_scraper import InternetScraper

def main():
    print("🌐 Universal Domain Database - Starting Up")
    print("=" * 50)
    
    # Initialize the self-learning system
    system = SelfLearningSystem()
    
    # Initialize the internet scraper
    scraper = InternetScraper()
    
    # Show system status
    status = system.get_system_status()
    print(f"📊 System Status:")
    print(f"   Domains: {status['total_domains']}")
    print(f"   Data Sources: {status['total_data_sources']}")
    print(f"   Learning Sessions: {status['learning_metrics']['learning_sessions']}")
    
    print("\n🚀 System ready for learning!")
    print("\nAvailable commands:")
    print("  python database/learning_engine/self_learning_system.py")
    print("  python database/scrapers/internet_scraper.py")
    print("  python database/start_learning.py")

if __name__ == "__main__":
    main()
