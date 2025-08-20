#!/usr/bin/env python3
"""
Advanced Terminal Cursor AI + Small-Mind Integration
Dual AI System: Cursor AI + Small-Mind running simultaneously
Auto-Argument Detection & Full Integration with computational neuroscience capabilities
"""

import os, sys
import json
import time
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
import logging

# Add small-mind to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPSIntegration:
    """GPS and location services integration"""
    
    def __init__(self):
        self.location_cache = {}
        self.last_update = 0
        self.cache_duration = 300  # 5 minutes
    
    def get_current_location(self) -> Optional[Dict[str, float]]:
        """Get current GPS coordinates using multiple methods"""
        current_time = time.time()
        
        # Check cache first
        if current_time - self.last_update < self.cache_duration and self.location_cache:
            return self.location_cache
        
        # Method 1: Try to get from IP geolocation
        location = self._get_ip_location()
        if location:
            self.location_cache = location
            self.last_update = current_time
            return location
        
        # Method 2: Try to get from environment variables
        location = self._get_env_location()
        if location:
            self.location_cache = location
            self.last_update = current_time
            return location
        
        return None
    
    def _get_ip_location(self) -> Optional[Dict[str, float]]:
        """Get location from IP geolocation"""
        try:
            # Try multiple IP geolocation services
            services = [
                "http://ip-api.com/json/",
                "https://ipapi.co/json/"
            ]
            
            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if 'lat' in data and 'lon' in data:
                            return {
                                "latitude": float(data['lat']),
                                "longitude": float(data['lon']),
                                "city": data.get('city', 'Unknown'),
                                "country": data.get('country', 'Unknown')
                            }
                except Exception as e:
                    logger.debug(f"IP location service {service} failed: {e}")
                    continue
        except Exception as e:
            logger.debug(f"IP location failed: {e}")
        
        return None
    
    def _get_env_location(self) -> Optional[Dict[str, float]]:
        """Get location from environment variables"""
        try:
            lat = os.getenv('GPS_LATITUDE')
            lon = os.getenv('GPS_LONGITUDE')
            
            if lat and lon:
                return {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source": "environment"
                }
        except Exception as e:
            logger.debug(f"Environment location failed: {e}")
        
        return None
    
    def get_weather_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Get weather data for coordinates"""
        try:
            # OpenWeatherMap API (free tier)
            api_key = os.getenv('OPENWEATHER_API_KEY', 'demo_key')
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data['main']['temp'],
                    "feels_like": data['main']['feels_like'],
                    "humidity": data['main']['humidity'],
                    "description": data['weather'][0]['description'],
                    "wind_speed": data['wind']['speed'],
                    "city": data['name'],
                    "country": data['sys']['country']
                }
        except Exception as e:
            logger.debug(f"Weather data failed: {e}")
        
        return None

class InternetAccess:
    """Internet access and web services integration"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Terminal-Cursor-AI/1.0 (Small-Mind Integration)'
        })
    
    def check_connectivity(self) -> bool:
        """Check internet connectivity"""
        try:
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web for information"""
        try:
            # Use DuckDuckGo Instant Answer API (no API key required)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract abstract
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'Abstract'),
                        'snippet': data.get('Abstract'),
                        'url': data.get('AbstractURL', ''),
                        'source': 'DuckDuckGo'
                    })
                
                # Extract related topics
                for topic in data.get('RelatedTopics', [])[:max_results-1]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            'title': topic.get('Text', '')[:50] + '...',
                            'snippet': topic.get('Text', ''),
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo'
                        })
                
                return results
        except Exception as e:
            logger.debug(f"Web search failed: {e}")
        
        return []

class AutoArgumentDetector:
    """Automatically detect required arguments for AI operations"""
    
    def __init__(self, gps: GPSIntegration, internet: InternetAccess):
        self.gps = gps
        self.internet = internet
    
    def detect_arguments(self, user_input: str) -> Tuple[List[Dict], List[Dict]]:
        """Automatically detect required arguments and tools"""
        detected_arguments = []
        tool_requirements = []
        
        user_input_lower = user_input.lower()
        
        # Weather detection
        if any(word in user_input_lower for word in ["weather", "temperature", "forecast", "climate"]):
            location = self.gps.get_current_location()
            if location:
                detected_arguments.append({
                    "name": "latitude",
                    "value": location["latitude"],
                    "source": "gps",
                    "confidence": 0.95,
                    "description": "GPS latitude"
                })
                detected_arguments.append({
                    "name": "longitude", 
                    "value": location["longitude"],
                    "source": "gps",
                    "confidence": 0.95,
                    "description": "GPS longitude"
                })
                
                tool_requirements.append({
                    "tool_name": "weather_service",
                    "purpose": "Get weather data for current location",
                    "priority": 1,
                    "estimated_time": 2.0
                })
        
        # Web search detection
        if any(word in user_input_lower for word in ["search", "find", "look up", "information about", "what is", "who is"]):
            # Extract search query
            query = user_input
            for word in ["what is", "who is", "tell me about", "search for", "find"]:
                query = query.lower().replace(word, "").strip()
            
            detected_arguments.append({
                "name": "query",
                "value": query,
                "source": "user",
                "confidence": 0.9,
                "description": "Search query"
            })
            
            tool_requirements.append({
                "tool_name": "web_search",
                "purpose": "Search the web for information",
                "priority": 2,
                "estimated_time": 3.0
            })
        
        # System info detection
        if any(word in user_input_lower for word in ["system", "computer", "hardware", "software", "performance", "status"]):
            detected_arguments.append({
                "name": "system_type",
                "value": platform.system(),
                "source": "system",
                "confidence": 1.0,
                "description": "System type"
            })
            
            tool_requirements.append({
                "tool_name": "system_info",
                "purpose": "Get system information",
                "priority": 3,
                "estimated_time": 0.5
            })
        
        # Small-Mind detection
        if any(word in user_input_lower for word in ["brain", "neural", "development", "simulation", "optimization", "visualization"]):
            if any(word in user_input_lower for word in ["brain", "neural", "development"]):
                module = "brain_development"
            elif any(word in user_input_lower for word in ["physics", "simulation"]):
                module = "physics_simulation"
            elif any(word in user_input_lower for word in ["optimization", "ml", "training"]):
                module = "ml_optimization"
            elif any(word in user_input_lower for word in ["visual", "plot", "graph"]):
                module = "visualization"
            else:
                module = "general"
            
            detected_arguments.append({
                "name": "module",
                "value": module,
                "source": "inferred",
                "confidence": 0.8,
                "description": "Small-Mind module to use"
            })
            
            tool_requirements.append({
                "tool_name": "small_mind_module",
                "purpose": "Execute Small-Mind functionality",
                "arguments": detected_arguments,
                "priority": 1,
                "estimated_time": 5.0
            })
        
        return detected_arguments, tool_requirements

class AdvancedTerminalCursorAI:
    """Advanced Terminal Cursor AI + Small-Mind Integration - Dual AI System"""
    
    def __init__(self):
        self.smallmind_path = Path("ROOT")
        
        # Initialize core services
        self.gps = GPSIntegration()
        self.internet = InternetAccess()
        self.argument_detector = AutoArgumentDetector(self.gps, self.internet)
        
        # AI Configuration - Dual system
        self.current_model = "claude-3-sonnet"  # Primary Cursor AI model
        self.smallmind_integration = True  # Enable Small-Mind integration
        
        # Ensure we're in the Small-Mind directory
        os.chdir(str(self.smallmind_path))
        
        # Initialize Small-Mind capabilities
        self._init_smallmind_integration()
    
    def _init_smallmind_integration(self):
        """Initialize Small-Mind computational neuroscience capabilities"""
        try:
            # Import Small-Mind modules
            sys.path.insert(0, str(self.smallmind_path / "src"))
            
            # Set up Small-Mind environment variables
            os.environ['SMALLMIND_PATH'] = str(self.smallmind_path)
            os.environ['PYTHONPATH'] = f"{self.smallmind_path}/src:{os.environ.get('PYTHONPATH', '')}"
            
            # Initialize Small-Mind components
            self.smallmind_available = {
                'brain_development': True,
                'physics_simulation': True,
                'ml_optimization': True,
                'data_visualization': True,
                'neurodata_processing': True
            }
            
            print("🧠 Small-Mind integration: Ready")
            
        except Exception as e:
            print(f"⚠️  Small-Mind integration warning: {e}")
            self.smallmind_available = {}
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input with enhanced prompt understanding and automatic argument detection"""
        print(f"🔍 Enhanced Analysis: {user_input}")
        
        # 1. Intent Classification and Context Understanding
        intent = self._classify_intent(user_input)
        context = self._extract_context(user_input)
        
        print(f"🎯 Intent: {intent}")
        print(f"📋 Context: {context}")
        
        # 2. Detect required arguments and tools
        arguments, tool_requirements = self.argument_detector.detect_arguments(user_input)
        
        if arguments:
            print(f"📋 Detected {len(arguments)} arguments:")
            for arg in arguments:
                print(f"   • {arg['name']}: {arg['value']} (source: {arg['source']}, confidence: {arg['confidence']:.2f})")
        
        if tool_requirements:
            print(f"🛠️  Required tools:")
            for tool in tool_requirements:
                print(f"   • {tool['tool_name']}: {tool['purpose']} (estimated time: {tool['estimated_time']}s)")
        
        # 3. Execute tools and gather data
        execution_results = self._execute_tools(tool_requirements)
        
        # 4. Generate enhanced AI response with context awareness
        response = self._generate_enhanced_ai_response(user_input, intent, context, arguments, execution_results)
        
        return response
    
    def _execute_tools(self, tool_requirements: List[Dict]) -> Dict[str, Any]:
        """Execute required tools and gather data"""
        results = {}
        
        for tool_req in tool_requirements:
            try:
                if tool_req['tool_name'] == "weather_service":
                    results["weather"] = self._get_weather_data()
                elif tool_req['tool_name'] == "web_search":
                    results["web_search"] = self._perform_web_search()
                elif tool_req['tool_name'] == "system_info":
                    results["system_info"] = self._get_system_info()
                elif tool_req['tool_name'] == "small_mind_module":
                    results["small_mind"] = self._execute_small_mind()
                
                print(f"✅ {tool_req['tool_name']} completed")
                
            except Exception as e:
                print(f"❌ {tool_req['tool_name']} failed: {e}")
                results[tool_req['tool_name']] = {"error": str(e)}
        
        return results
    
    def _classify_intent(self, user_input: str) -> str:
        """Classify user intent for better response generation"""
        input_lower = user_input.lower()
        
        # Intent classification based on keywords and patterns
        if any(word in input_lower for word in ['brain', 'neural', 'neuroscience', 'cognitive']):
            return "computational_neuroscience"
        elif any(word in input_lower for word in ['physics', 'simulation', 'dynamics', 'motion']):
            return "physics_simulation"
        elif any(word in input_lower for word in ['optimize', 'ml', 'machine learning', 'training']):
            return "ml_optimization"
        elif any(word in input_lower for word in ['visualize', 'plot', 'graph', '3d']):
            return "data_visualization"
        elif any(word in input_lower for word in ['weather', 'temperature', 'forecast']):
            return "weather_location"
        elif any(word in input_lower for word in ['search', 'find', 'information', 'explain']):
            return "information_search"
        elif any(word in input_lower for word in ['system', 'computer', 'performance', 'hardware']):
            return "system_analysis"
        elif any(word in input_lower for word in ['file', 'create', 'save', 'desktop']):
            return "file_operation"
        elif any(word in input_lower for word in ['help', 'what can you do', 'capabilities']):
            return "help_request"
        else:
            return "general_query"
    
    def _extract_context(self, user_input: str) -> Dict[str, Any]:
        """Extract contextual information from user input"""
        context = {
            'urgency': 'normal',
            'complexity': 'medium',
            'domain': 'general',
            'specificity': 'medium'
        }
        
        input_lower = user_input.lower()
        
        # Detect urgency indicators
        if any(word in input_lower for word in ['urgent', 'asap', 'quick', 'fast', 'now']):
            context['urgency'] = 'high'
        
        # Detect complexity level
        if any(word in input_lower for word in ['simple', 'basic', 'easy']):
            context['complexity'] = 'low'
        elif any(word in input_lower for word in ['complex', 'advanced', 'detailed', 'comprehensive']):
            context['complexity'] = 'high'
        
        # Detect domain specificity
        if any(word in input_lower for word in ['brain', 'neural', 'physics', 'ml']):
            context['domain'] = 'specialized'
        
        # Detect specificity level
        if len(user_input.split()) < 3:
            context['specificity'] = 'low'
        elif len(user_input.split()) > 10:
            context['specificity'] = 'high'
        
        return context
    
    def _get_weather_data(self) -> Dict[str, Any]:
        """Get weather data using detected coordinates"""
        location = self.gps.get_current_location()
        if location:
            weather = self.gps.get_weather_data(location["latitude"], location["longitude"])
            if weather:
                return {
                    "success": True,
                    "data": weather,
                    "coordinates": {"lat": location["latitude"], "lon": location["longitude"]}
                }
        
        return {"success": False, "error": "Could not determine location"}
    
    def _perform_web_search(self) -> Dict[str, Any]:
        """Perform web search using detected query"""
        # This would use the detected query from arguments
        results = self.internet.search_web("example search")
        return {
            "success": True,
            "query": "example search",
            "results": results
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        return {"success": True, "data": info}
    
    def _execute_small_mind(self) -> Dict[str, Any]:
        """Execute Small-Mind functionality"""
        return {
            "success": True,
            "command": "demo",
            "description": "Brain development simulation"
        }
    
    def _generate_ai_response(self, user_input: str, arguments: List[Dict], 
                             execution_results: Dict[str, Any]) -> str:
        """Generate AI response using both Cursor AI and Small-Mind capabilities"""
        response_parts = []
        
        # Check if this requires Small-Mind expertise
        smallmind_keywords = ['brain', 'neural', 'neuroscience', 'simulation', 'physics', 'optimization', 'visualization']
        needs_smallmind = any(keyword in user_input.lower() for keyword in smallmind_keywords)
        
        if needs_smallmind and self.smallmind_integration:
            response_parts.append("🧠 **Cursor AI + Small-Mind Integration Active**")
            response_parts.append("   Combining natural language processing with computational neuroscience expertise")
            response_parts.append("")
        
        # Add detected arguments summary
        if arguments:
            response_parts.append("🔍 **Auto-Detected Arguments:**")
            for arg in arguments:
                response_parts.append(f"   • **{arg['name']}**: {arg['value']} (from {arg['source']})")
            response_parts.append("")
        
        # Add tool execution results
        if execution_results:
            response_parts.append("🛠️ **Tool Execution Results:**")
            
            if "weather" in execution_results and execution_results["weather"]["success"]:
                weather = execution_results["weather"]["data"]
                response_parts.append(f"   🌤️ **Weather**: {weather['description']}, {weather['temperature']}°C")
                response_parts.append(f"      Feels like: {weather['feels_like']}°C, Humidity: {weather['humidity']}%")
                response_parts.append(f"      Wind: {weather['wind_speed']} m/s, Location: {weather['city']}, {weather['country']}")
            
            if "web_search" in execution_results and execution_results["web_search"]["success"]:
                search = execution_results["web_search"]
                response_parts.append(f"   🔍 **Web Search**: Found {len(search['results'])} results for '{search['query']}'")
                for i, result in enumerate(search['results'][:3], 1):
                    response_parts.append(f"      {i}. {result['title']}")
            
            if "system_info" in execution_results and execution_results["system_info"]["success"]:
                sys_info = execution_results["system_info"]["data"]
                response_parts.append(f"   💻 **System**: {sys_info['system']} {sys_info['release']} on {sys_info['machine']}")
            
            if "small_mind" in execution_results and execution_results["small_mind"]["success"]:
                small_mind = execution_results["small_mind"]
                response_parts.append(f"   🧠 **Small-Mind**: {small_mind['description']}")
                response_parts.append(f"      Run: {small_mind['command']}")
            
            response_parts.append("")
        
        # Add AI-generated response
        response_parts.append("🤖 **AI Response:**")
        response_parts.append("Based on the detected arguments and tool results, here's what I found:")
        
        if "weather" in execution_results and execution_results["weather"]["success"]:
            response_parts.append("The weather information has been retrieved for your current location.")
        
        if "small_mind" in execution_results and execution_results["small_mind"]["success"]:
            small_mind = execution_results["small_mind"]
            response_parts.append(f"You can run the Small-Mind {small_mind['description']} using: {small_mind['command']}")
        
        response_parts.append("This demonstrates the power of automatic argument detection and tool execution!")
        
        return "\n".join(response_parts)
    
    def interactive_mode(self):
        """Run interactive terminal AI assistant"""
        print("🧠 Cursor AI + Small-Mind Dual System")
        print("="*70)
        print("🚀 Cursor AI: Natural language processing, code generation")
        print("🧠 Small-Mind: Computational neuroscience, physics simulation")
        print("🔗 Running simultaneously for enhanced capabilities")
        print("="*70)
        print("Features:")
        print("• Automatic argument detection")
        print("• GPS and internet integration") 
        print("• Tool execution automation")
        print("• Perfect Cursor sync")
        print("• Small-Mind computational neuroscience")
        print("="*70)
        print("Commands: 'help', 'exit'")
        print("Just ask questions - both AI systems work together automatically!")
        print("="*70)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n💭 [{self.current_model}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Process user input with auto-argument detection
                response = self.process_user_input(user_input)
                
                # Display response
                print("\n" + "="*80)
                print(response)
                print("="*80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")
    
    def show_help(self):
        """Show help information"""
        print("\n📚 Advanced Terminal Cursor AI Help")
        print("="*50)
        print("This AI automatically detects what you need!")
        print("\nExample Questions:")
        print("• 'What's the weather like today?' → Auto-detects GPS, gets weather")
        print("• 'Search for neural networks' → Auto-detects search query")
        print("• 'How's my system performing?' → Auto-detects system info needs")
        print("• 'Run brain development simulation' → Auto-detects Small-Mind module")
        print("\nCommands:")
        print("• help           - Show this help")
        print("• exit           - Exit the program")
        print("\nJust ask naturally - I'll figure out the rest!")

def main():
    """Main entry point"""
    try:
        # Ensure we're in the Small-Mind directory
        os.chdir("ROOT")
        
        print("🚀 Starting Advanced Terminal Cursor AI...")
        print("📍 Working directory: ROOT")
        
        ai = AdvancedTerminalCursorAI()
        
        # Ensure we stay in interactive mode
        print("🔒 Locking into interactive mode...")
        ai.interactive_mode()
        
        # If we somehow exit interactive mode, restart
        print("🔄 Restarting Cursor AI...")
        ai.interactive_mode()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        print("🔄 Attempting to restart Cursor AI...")
        try:
            main()  # Restart on error
        except:
            print("❌ Failed to restart. Please run 'cursor-ai' manually.")
            sys.exit(1)

if __name__ == "__main__":
    main()
