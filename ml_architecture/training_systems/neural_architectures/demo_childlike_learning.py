#!/usr/bin/env python3
"""
Demo: Childlike Learning System for Llama Lexi
Showcases how the system mimics 5-year-old language learning
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.childlike_training import ChildlikeTrainer
from models.child_learner import ChildLearner

def demo_basic_learning():
    """Demonstrate basic childlike learning"""
    print("🧒 Demo: Basic Childlike Learning")
    print("=" * 50)
    
    # Create a child learner
    learner = ChildLearner()
    
    # Discover some words naturally
    print("\n1. Discovering New Words (like a child hearing them for the first time)")
    learner.discover_word("butterfly", "in the garden", "excited")
    learner.discover_word("thunder", "during a storm", "scared")
    learner.discover_word("rainbow", "after the rain", "happy")
    
    # Ask childlike questions
    print("\n2. Asking Childlike Questions (the natural 'why?' phase)")
    for word in ["butterfly", "thunder", "rainbow"]:
        question = learner.ask_question(word)
        print(f"   🤔 {question}")
    
    # Explore contexts
    print("\n3. Exploring Contexts (applying words in new situations)")
    learner.explore_context("butterfly", "flying in the sky")
    learner.explore_context("rainbow", "in a storybook")
    learner.explore_context("thunder", "in a scary movie")
    
    # Find connections
    print("\n4. Finding Connections (children naturally connect things)")
    for word in ["butterfly", "rainbow"]:
        connections = learner.find_connections(word)
        if connections:
            print(f"\n   🔗 Connections for '{word}':")
            for connection in connections:
                print(f"      {connection}")
    
    # Show learning status
    print("\n5. Learning Progress (tracking natural development)")
    status = learner.get_learning_status()
    print(f"   📊 Total words: {status['total_words']}")
    print(f"   📊 Mastered words: {status['mastered_words']}")
    print(f"   📊 Learning progress: {status['learning_progress']:.1%}")
    
    return learner

def demo_learning_scenarios():
    """Demonstrate learning through scenarios"""
    print("\n\n🎯 Demo: Learning Through Scenarios")
    print("=" * 50)
    
    # Create trainer
    trainer = ChildlikeTrainer()
    
    # Show available scenarios
    print("\nAvailable Learning Scenarios:")
    trainer._show_scenarios()
    
    # Run a guided session
    print("\nRunning a guided learning session...")
    session_words = trainer.start_learning_session(duration_minutes=10)
    
    return trainer

def demo_interactive_learning():
    """Demonstrate interactive learning mode"""
    print("\n\n🎮 Demo: Interactive Learning Mode")
    print("=" * 50)
    
    print("This mode lets you guide the learning process like a parent or teacher.")
    print("You can discover words, ask questions, and explore contexts.")
    
    # Create trainer
    trainer = ChildlikeTrainer()
    
    # Show some example interactions
    print("\nExample interactions:")
    print("   discover rainbow")
    print("   ask butterfly")
    print("   explore thunder")
    print("   connect flower")
    
    print("\nTo try interactive mode, run:")
    print("   python childlike_training.py")
    print("   Then choose option 2 (Interactive Learning)")
    
    return trainer

def demo_curiosity_patterns():
    """Demonstrate curiosity-driven learning patterns"""
    print("\n\n🔍 Demo: Curiosity-Driven Learning Patterns")
    print("=" * 50)
    
    learner = ChildLearner()
    
    # Show how curiosity changes over time
    print("1. Initial Discovery (high curiosity)")
    learner.discover_word("dragon", "in a storybook", "excited")
    
    print("\n2. Repetition (curiosity maintained)")
    learner.discover_word("dragon", "in a movie", "excited")
    learner.discover_word("dragon", "in a game", "excited")
    
    print("\n3. Context Exploration (curiosity boosted)")
    learner.explore_context("dragon", "flying in the sky")
    learner.explore_context("dragon", "breathing fire")
    
    # Show word data
    word_data = learner.known_words["dragon"]
    print(f"\n4. Word Learning Data:")
    print(f"   Times heard: {word_data['times_heard']}")
    print(f"   Contexts: {word_data['contexts']}")
    print(f"   Curiosity level: {word_data['curiosity']:.2f}")
    print(f"   Questions asked: {len(word_data['questions_asked'])}")
    
    return learner

def demo_emotional_learning():
    """Demonstrate emotional connection to language"""
    print("\n\n😊 Demo: Emotional Connection to Language")
    print("=" * 50)
    
    learner = ChildLearner()
    
    # Discover words with different emotions
    print("1. Happy Words")
    learner.discover_word("birthday", "celebration", "happy")
    learner.discover_word("puppy", "playing together", "excited")
    learner.discover_word("ice cream", "summer treat", "joyful")
    
    print("\n2. Scary Words")
    learner.discover_word("monster", "under the bed", "scared")
    learner.discover_word("dark", "night time", "nervous")
    learner.discover_word("loud", "sudden noise", "startled")
    
    print("\n3. Sad Words")
    learner.discover_word("goodbye", "friend leaving", "sad")
    learner.discover_word("broken", "favorite toy", "disappointed")
    learner.discover_word("lost", "can't find something", "worried")
    
    # Show emotional patterns
    print("\n4. Emotional Learning Patterns:")
    for word, data in learner.known_words.items():
        emotions = ", ".join(data['emotions'])
        print(f"   '{word}': {emotions}")
    
    return learner

def main():
    """Run all demos"""
    print("🧒 Childlike Learning System - Complete Demo")
    print("=" * 60)
    print("This demo shows how AI can learn language like a 5-year-old")
    print("Through curiosity, play, and natural discovery")
    print("=" * 60)
    
    try:
        # Run all demos
        learner1 = demo_basic_learning()
        trainer = demo_learning_scenarios()
        trainer2 = demo_interactive_learning()
        learner2 = demo_curiosity_patterns()
        learner3 = demo_emotional_learning()
        
        print("\n\n🎉 Demo Complete!")
        print("=" * 60)
        print("What you've seen:")
        print("   ✅ Natural word discovery")
        print("   ✅ Childlike question generation")
        print("   ✅ Context exploration")
        print("   ✅ Connection building")
        print("   ✅ Curiosity-driven learning")
        print("   ✅ Emotional language connection")
        print("   ✅ Scenario-based learning")
        print("   ✅ Progress tracking")
        
        print("\n🚀 Next Steps:")
        print("   1. Run interactive learning: python childlike_training.py")
        print("   2. Explore custom scenarios")
        print("   3. Integrate with Llama Lexi model")
        print("   4. Create your own learning paths")
        
        # Save progress from all learners
        print("\n💾 Saving demo progress...")
        learner1.save_progress("demo_basic_learning.json")
        learner2.save_progress("demo_curiosity_patterns.json")
        learner3.save_progress("demo_emotional_learning.json")
        print("   Progress saved to demo files")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements_childlike_learning.txt")

if __name__ == "__main__":
    main()
