#!/usr/bin/env python3
"""
Redundancy Detection Module
===========================

Detects both exact duplicates and redundant files with similar:
- Functionality (similar purpose, overlapping features)
- Content (similar text, code patterns, configurations)
- Structure (similar file organization, naming patterns)
- Dependencies (importing similar modules, using similar libraries)

Author: Quark Brain Architecture
Date: 2024
"""

import os
import re
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import time

class RedundancyDetector:
    """Detects redundant files beyond just exact duplicates"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.similarity_threshold = 0.75  # 75% similarity = redundant
        
    def find_functional_redundancies(self) -> List[Tuple[str, str, float, str]]:
        """Find files with similar functionality/purpose"""
        print("🔍 Analyzing functional redundancies...")
        
        # Group files by functionality patterns
        functional_groups = defaultdict(list)
        
        for file_path in self.repo_root.rglob("*.py"):
            if file_path.is_file():
                functionality = self._extract_functionality(file_path)
                if functionality:
                    functional_groups[functionality].append(str(file_path))
        
        # Find groups with multiple files (potential redundancies)
        redundancies = []
        for functionality, files in functional_groups.items():
            if len(files) > 1:
                # Analyze similarity within the group
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        similarity = self._calculate_functional_similarity(file1, file2)
                        if similarity > self.similarity_threshold:
                            redundancies.append((
                                file1, file2, similarity, 
                                f"Similar functionality: {functionality}"
                            ))
        
        print(f"📋 Found {len(redundancies)} functional redundancies")
        return redundancies
    
    def _extract_functionality(self, file_path: Path) -> str:
        """Extract functionality description from file"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Look for function/class names that indicate purpose
            functionality_indicators = []
            
            # Function definitions
            func_matches = re.findall(r'def\s+(\w+)\s*\(', content)
            functionality_indicators.extend(func_matches)
            
            # Class definitions
            class_matches = re.findall(r'class\s+(\w+)', content)
            functionality_indicators.extend(class_matches)
            
            # Import statements (indicate purpose)
            import_matches = re.findall(r'from\s+(\w+)\s+import|import\s+(\w+)', content)
            for match in import_matches:
                functionality_indicators.extend([m for m in match if m])
            
            # File naming patterns
            filename = file_path.stem.lower()
            if 'test' in filename:
                functionality_indicators.append('testing')
            if 'util' in filename or 'helper' in filename:
                functionality_indicators.append('utility')
            if 'config' in filename or 'settings' in filename:
                functionality_indicators.append('configuration')
            
            # Combine indicators
            if functionality_indicators:
                return '|'.join(sorted(set(functionality_indicators)))
            
        except Exception:
            pass
        
        return ""
    
    def _calculate_functional_similarity(self, file1: str, file2: str) -> float:
        """Calculate functional similarity between two files"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()
            
            # Use difflib for content similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity
            
        except Exception:
            return 0.0
    
    def find_content_redundancies(self) -> List[Tuple[str, str, float, str]]:
        """Find files with similar content patterns"""
        print("🔍 Analyzing content redundancies...")
        
        # Group files by content patterns
        content_groups = defaultdict(list)
        
        for file_path in self.repo_root.rglob("*.py"):
            if file_path.is_file():
                content_pattern = self._extract_content_pattern(file_path)
                if content_pattern:
                    content_groups[content_pattern].append(str(file_path))
        
        # Find groups with similar content
        redundancies = []
        for pattern, files in content_groups.items():
            if len(files) > 1:
                # Analyze similarity within the group
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        similarity = self._calculate_content_similarity(file1, file2)
                        if similarity > self.similarity_threshold:
                            redundancies.append((
                                file1, file2, similarity,
                                f"Similar content pattern: {pattern[:50]}..."
                            ))
        
        print(f"📋 Found {len(redundancies)} content redundancies")
        return redundancies
    
    def _extract_content_pattern(self, file_path: Path) -> str:
        """Extract content pattern signature"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Extract key content patterns
            patterns = []
            
            # Function/class density
            func_count = len(re.findall(r'def\s+\w+', content))
            class_count = len(re.findall(r'class\s+\w+', content))
            patterns.append(f"funcs:{func_count}")
            patterns.append(f"classes:{class_count}")
            
            # Import patterns
            imports = re.findall(r'import\s+(\w+)|from\s+(\w+)\s+import', content)
            import_modules = set()
            for match in imports:
                import_modules.update([m for m in match if m])
            if import_modules:
                patterns.append(f"imports:{','.join(sorted(import_modules))}")
            
            # Code structure patterns
            lines = content.split('\n')
            empty_lines = len([l for l in lines if not l.strip()])
            comment_lines = len([l for l in lines if l.strip().startswith('#')])
            patterns.append(f"empty:{empty_lines}")
            patterns.append(f"comments:{comment_lines}")
            
            return '|'.join(patterns)
            
        except Exception:
            pass
        
        return ""
    
    def _calculate_content_similarity(self, file1: str, file2: str) -> float:
        """Calculate content similarity between two files"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()
            
            # Use difflib for content similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity
            
        except Exception:
            return 0.0
    
    def find_structural_redundancies(self) -> List[Tuple[str, str, float, str]]:
        """Find files with similar structure/organization"""
        print("🔍 Analyzing structural redundancies...")
        
        # Group files by structural patterns
        structural_groups = defaultdict(list)
        
        for file_path in self.repo_root.rglob("*.py"):
            if file_path.is_file():
                structure = self._extract_structure(file_path)
                if structure:
                    structural_groups[structure].append(str(file_path))
        
        # Find groups with similar structure
        redundancies = []
        for structure, files in structural_groups.items():
            if len(files) > 1:
                # Analyze similarity within the group
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        similarity = self._calculate_structural_similarity(file1, file2)
                        if similarity > self.similarity_threshold:
                            redundancies.append((
                                file1, file2, similarity,
                                f"Similar structure: {structure[:50]}..."
                            ))
        
        print(f"📋 Found {len(redundancies)} structural redundancies")
        return redundancies
    
    def _extract_structure(self, file_path: Path) -> str:
        """Extract structural signature"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Extract structural patterns
            structure = []
            
            # File organization
            lines = content.split('\n')
            structure.append(f"total_lines:{len(lines)}")
            
            # Indentation patterns
            indent_levels = []
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    indent_levels.append(indent)
            
            if indent_levels:
                avg_indent = sum(indent_levels) / len(indent_levels)
                structure.append(f"avg_indent:{avg_indent:.1f}")
            
            # Function/class organization
            func_positions = []
            class_positions = []
            
            for i, line in enumerate(lines):
                if re.match(r'def\s+\w+', line.strip()):
                    func_positions.append(i)
                elif re.match(r'class\s+\w+', line.strip()):
                    class_positions.append(i)
            
            structure.append(f"funcs_at:{','.join(map(str, func_positions[:5]))}")
            structure.append(f"classes_at:{','.join(map(str, class_positions[:5]))}")
            
            return '|'.join(structure)
            
        except Exception:
            pass
        
        return ""
    
    def _calculate_structural_similarity(self, file1: str, file2: str) -> float:
        """Calculate structural similarity between two files"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()
            
            # Use difflib for structural similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity
            
        except Exception:
            return 0.0
    
    def find_dependency_redundancies(self) -> List[Tuple[str, str, float, str]]:
        """Find files with similar dependencies"""
        print("🔍 Analyzing dependency redundancies...")
        
        # Group files by dependency patterns
        dependency_groups = defaultdict(list)
        
        for file_path in self.repo_root.rglob("*.py"):
            if file_path.is_file():
                dependencies = self._extract_dependencies(file_path)
                if dependencies:
                    dependency_groups[dependencies].append(str(file_path))
        
        # Find groups with similar dependencies
        redundancies = []
        for deps, files in dependency_groups.items():
            if len(files) > 1:
                # Analyze similarity within the group
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        similarity = self._calculate_dependency_similarity(file1, file2)
                        if similarity > self.similarity_threshold:
                            redundancies.append((
                                file1, file2, similarity,
                                f"Similar dependencies: {deps[:50]}..."
                            ))
        
        print(f"📋 Found {len(redundancies)} dependency redundancies")
        return redundancies
    
    def _extract_dependencies(self, file_path: Path) -> str:
        """Extract dependency signature"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Extract import statements
            imports = re.findall(r'import\s+(\w+)|from\s+(\w+)\s+import', content)
            import_modules = set()
            for match in imports:
                import_modules.update([m for m in match if m])
            
            # Extract function calls
            function_calls = re.findall(r'(\w+)\s*\(', content)
            function_calls = [f for f in function_calls if f not in ['if', 'for', 'while', 'def', 'class']]
            
            # Combine dependencies
            dependencies = []
            if import_modules:
                dependencies.append(f"imports:{','.join(sorted(import_modules))}")
            if function_calls:
                dependencies.append(f"calls:{','.join(sorted(set(function_calls))[:10])}")
            
            return '|'.join(dependencies)
            
        except Exception:
            pass
        
        return ""
    
    def _calculate_dependency_similarity(self, file1: str, file2: str) -> float:
        """Calculate dependency similarity between two files"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()
            
            # Use difflib for dependency similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity
            
        except Exception:
            return 0.0
    
    def find_all_redundancies(self) -> Dict[str, List[Tuple[str, str, float, str]]]:
        """Find all types of redundancies"""
        print("🧠 Comprehensive redundancy analysis...")
        start_time = time.time()
        
        all_redundancies = {
            'functional': self.find_functional_redundancies(),
            'content': self.find_content_redundancies(),
            'structural': self.find_structural_redundancies(),
            'dependency': self.find_dependency_redundancies()
        }
        
        total_redundancies = sum(len(redundancies) for redundancies in all_redundancies.values())
        elapsed_time = time.time() - start_time
        
        print(f"✅ Redundancy analysis complete in {elapsed_time:.2f}s")
        print(f"📊 Total redundancies found: {total_redundancies}")
        
        for redundancy_type, redundancies in all_redundancies.items():
            print(f"   {redundancy_type.capitalize()}: {len(redundancies)}")
        
        return all_redundancies
    
    def generate_redundancy_report(self, redundancies: Dict[str, List[Tuple[str, str, float, str]]]) -> str:
        """Generate comprehensive redundancy report"""
        report = f"""
🔍 COMPREHENSIVE REDUNDANCY ANALYSIS REPORT
==========================================

📊 Summary:
"""
        
        total_redundancies = sum(len(redundancies) for redundancies in redundancies.values())
        report += f"   Total Redundancies: {total_redundancies}\n"
        
        for redundancy_type, type_redundancies in redundancies.items():
            report += f"   {redundancy_type.capitalize()}: {len(type_redundancies)}\n"
        
        report += "\n🎯 Detailed Analysis:\n"
        
        for redundancy_type, type_redundancies in redundancies.items():
            if type_redundancies:
                report += f"\n📁 {redundancy_type.upper()} REDUNDANCIES:\n"
                report += "-" * 50 + "\n"
                
                # Sort by similarity score
                sorted_redundancies = sorted(type_redundancies, key=lambda x: x[2], reverse=True)
                
                for file1, file2, similarity, reason in sorted_redundancies[:10]:  # Show top 10
                    report += f"Similarity: {similarity:.2%}\n"
                    report += f"File 1: {file1}\n"
                    report += f"File 2: {file2}\n"
                    report += f"Reason: {reason}\n"
                    report += "-" * 30 + "\n"
                
                if len(type_redundancies) > 10:
                    report += f"... and {len(type_redundancies) - 10} more\n"
        
        return report

def main():
    """Test the redundancy detector"""
    detector = RedundancyDetector('.')
    
    print("🧠 Starting comprehensive redundancy analysis...")
    redundancies = detector.find_all_redundancies()
    
    # Generate report
    report = detector.generate_redundancy_report(redundancies)
    print(report)
    
    # Save report to file
    with open('redundancy_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("📄 Report saved to 'redundancy_analysis_report.txt'")

if __name__ == "__main__":
    main()
