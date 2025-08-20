#!/usr/bin/env python3
"""
Update Import Paths Script
Updates Python import statements to match the new Brain-ML Synergy Architecture

This script scans all Python files in the new architecture and updates import statements
to reflect the new directory structure.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class ImportPathUpdater:
    def __init__(self):
        self.new_structure = {
            # Brain Architecture
            'brain_modules': '🧠_BRAIN_ARCHITECTURE/01_NEURAL_CORE',
            'brain_hierarchy': '🧠_BRAIN_ARCHITECTURE/02_BRAIN_HIERARCHY',
            
            # ML Architecture
            'expert_domains': '🤖_ML_ARCHITECTURE/01_EXPERT_DOMAINS',
            'training': '🤖_ML_ARCHITECTURE/02_TRAINING_SYSTEMS',
            'knowledge_systems': '🤖_ML_ARCHITECTURE/03_KNOWLEDGE_SYSTEMS',
            
            # Integration
            'architecture': '🔄_INTEGRATION/01_ARCHITECTURE',
            'applications': '🔄_INTEGRATION/02_APPLICATIONS',
            
            # Data & Knowledge
            'data': '📊_DATA_KNOWLEDGE/01_DATA_REPOSITORY',
            'models': '📊_DATA_KNOWLEDGE/02_MODELS_ARTIFACTS',
            'research_lab': '📊_DATA_KNOWLEDGE/03_RESEARCH',
            
            # Development
            'development_stages': '🛠️_DEVELOPMENT/01_DEVELOPMENT_STAGES',
            'tools_utilities': '🛠️_DEVELOPMENT/02_TOOLS_UTILITIES',
            'deployment': '🛠️_DEVELOPMENT/03_DEPLOYMENT',
            
            # Management
            'configs': '📋_MANAGEMENT/01_CONFIGURATIONS',
            'project_management': '📋_MANAGEMENT/02_PROJECT_MANAGEMENT',
            
            # Testing
            'tests': '🧪_TESTING/01_TESTING_FRAMEWORKS',
            'results': '🧪_TESTING/02_RESULTS_OUTPUTS',
            
            # Documentation
            'docs': '📚_DOCUMENTATION/01_DOCS',
            'summaries': '📚_DOCUMENTATION/02_SUMMARIES',
            'reports': '📚_DOCUMENTATION/03_REPORTS'
        }
        
        self.import_mappings = self._create_import_mappings()
        
    def _create_import_mappings(self) -> Dict[str, str]:
        """Create mappings from old import paths to new ones"""
        mappings = {}
        
        # Brain modules specific mappings
        mappings['from brain_architecture.neural_core'] = 'from 🧠_BRAIN_ARCHITECTURE.01_NEURAL_CORE'
        mappings['import brain_architecture.neural_core'] = 'import 🧠_BRAIN_ARCHITECTURE.01_NEURAL_CORE'
        
        # Expert domains mappings
        mappings['from ml_architecture.expert_domains'] = 'from 🤖_ML_ARCHITECTURE.01_EXPERT_DOMAINS'
        mappings['import ml_architecture.expert_domains'] = 'import 🤖_ML_ARCHITECTURE.01_EXPERT_DOMAINS'
        
        # Training mappings
        mappings['from ml_architecture.training_pipelines'] = 'from 🤖_ML_ARCHITECTURE.02_TRAINING_SYSTEMS'
        mappings['import ml_architecture.training_pipelines'] = 'import 🤖_ML_ARCHITECTURE.02_TRAINING_SYSTEMS'
        
        # Knowledge systems mappings
        mappings['from data_knowledge.datasets_knowledge.datasets_knowledge.knowledge_systems'] = 'from 🤖_ML_ARCHITECTURE.03_KNOWLEDGE_SYSTEMS'
        mappings['import data_knowledge.datasets_knowledge.datasets_knowledge.knowledge_systems'] = 'import 🤖_ML_ARCHITECTURE.03_KNOWLEDGE_SYSTEMS'
        
        # Architecture mappings
        mappings['from integration.architecture'] = 'from 🔄_INTEGRATION.01_ARCHITECTURE'
        mappings['import integration.architecture'] = 'import 🔄_INTEGRATION.01_ARCHITECTURE'
        
        # Data mappings
        mappings['from data_knowledge.datasets_knowledge.datasets_knowledge.datasets'] = 'from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORY'
        mappings['import data_knowledge.datasets_knowledge.datasets_knowledge.datasets'] = 'import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORY'
        
        # Tools mappings
        mappings['from development.tools_utilities'] = 'from 🛠️_DEVELOPMENT.02_TOOLS_UTILITIES'
        mappings['import development.tools_utilities'] = 'import 🛠️_DEVELOPMENT.02_TOOLS_UTILITIES'
        
        # Testing mappings
        mappings['from testing.test_suites'] = 'from 🧪_TESTING.01_TESTING_FRAMEWORKS'
        mappings['import testing.test_suites'] = 'import 🧪_TESTING.01_TESTING_FRAMEWORKS'
        
        return mappings
    
    def find_python_files(self, root_dir: str) -> List[str]:
        """Find all Python files in the new architecture"""
        python_files = []
        
        for root, dirs, files in os.walk(root_dir):
            # Skip hidden directories and specific patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def update_imports_in_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """Update import statements in a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = []
            
            # Apply import mappings
            for old_import, new_import in self.import_mappings.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    changes_made.append(f"Updated: {old_import} → {new_import}")
            
            # Update relative imports for new structure
            content = self._update_relative_imports(content, file_path)
            
            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, changes_made
            
            return False, []
            
        except Exception as e:
            return False, [f"Error processing {file_path}: {str(e)}"]
    
    def _update_relative_imports(self, content: str, file_path: str) -> str:
        """Update relative imports to work with new structure"""
        # This is a simplified version - in practice, you'd need more sophisticated
        # logic to handle relative imports based on the new directory structure
        
        # Example: update common relative import patterns
        content = re.sub(r'from \.\.', 'from .............................................................', content)
        content = re.sub(r'from \.', 'from ...........................................................', content)
        
        return content
    
    def update_all_imports(self, root_dir: str = '.') -> Dict[str, List[str]]:
        """Update imports in all Python files"""
        print("🔍 Scanning for Python files...")
        python_files = self.find_python_files(root_dir)
        
        print(f"📁 Found {len(python_files)} Python files")
        
        results = {
            'updated': [],
            'errors': [],
            'no_changes': []
        }
        
        for file_path in python_files:
            print(f"🔄 Processing: {file_path}")
            
            try:
                updated, changes = self.update_imports_in_file(file_path)
                
                if updated:
                    results['updated'].append({
                        'file': file_path,
                        'changes': changes
                    })
                    print(f"✅ Updated: {file_path}")
                else:
                    results['no_changes'].append(file_path)
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        return results
    
    def generate_import_report(self, results: Dict[str, List[str]]) -> str:
        """Generate a report of import updates"""
        report = "# 📊 Import Path Update Report\n\n"
        
        report += f"## Summary\n"
        report += f"- **Files Updated**: {len(results['updated'])}\n"
        report += f"- **Files Unchanged**: {len(results['no_changes'])}\n"
        report += f"- **Errors**: {len(results['errors'])}\n\n"
        
        if results['updated']:
            report += "## Updated Files\n\n"
            for item in results['updated']:
                report += f"### {item['file']}\n"
                for change in item['changes']:
                    report += f"- {change}\n"
                report += "\n"
        
        if results['errors']:
            report += "## Errors\n\n"
            for error in results['errors']:
                report += f"- {error}\n"
        
        return report

def main():
    """Main execution function"""
    print("🧠🤖 Brain-ML Synergy Architecture: Import Path Updater")
    print("=" * 60)
    
    updater = ImportPathUpdater()
    
    # Update imports in the new architecture directories
    new_arch_dirs = [
        '🧠_BRAIN_ARCHITECTURE',
        '🤖_ML_ARCHITECTURE', 
        '🔄_INTEGRATION',
        '📊_DATA_KNOWLEDGE',
        '🛠️_DEVELOPMENT',
        '📋_MANAGEMENT',
        '🧪_TESTING',
        '📚_DOCUMENTATION'
    ]
    
    all_results = {
        'updated': [],
        'errors': [],
        'no_changes': []
    }
    
    for arch_dir in new_arch_dirs:
        if os.path.exists(arch_dir):
            print(f"\n🏗️ Processing {arch_dir}...")
            results = updater.update_all_imports(arch_dir)
            
            # Merge results
            all_results['updated'].extend(results['updated'])
            all_results['errors'].extend(results['errors'])
            all_results['no_changes'].extend(results['no_changes'])
    
    # Generate and save report
    report = updater.generate_import_report(all_results)
    
    with open('IMPORT_UPDATE_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Import Update Complete!")
    print(f"✅ Files Updated: {len(all_results['updated'])}")
    print(f"⚠️ Errors: {len(all_results['errors'])}")
    print(f"📄 Report saved to: IMPORT_UPDATE_REPORT.md")

if __name__ == "__main__":
    main()
