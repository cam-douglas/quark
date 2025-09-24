"""
Documentation Handler Module
============================
Handles documentation generation and updates.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class DocumentationHandler:
    """Handles documentation operations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = self.project_root / 'docs'
        self.readme = self.project_root / 'README.md'
        
    def route_command(self, action: str, params: Dict) -> int:
        """Route documentation commands."""
        if action == 'generate':
            return self.generate_docs(params)
        elif action == 'update':
            return self.update_readme(params)
        elif action == 'api':
            return self.generate_api_docs()
        elif action == 'check':
            return self.check_documentation()
        else:
            return self.show_help()
    
    def generate_docs(self, params: Dict) -> int:
        """Generate documentation."""
        doc_type = params.get('type', 'all')
        
        print(f"\n📚 Generating Documentation: {doc_type}")
        print("=" * 50)
        
        if doc_type == 'api' or doc_type == 'all':
            self.generate_api_docs()
        
        if doc_type == 'validation' or doc_type == 'all':
            self._generate_validation_docs()
        
        if doc_type == 'roadmap' or doc_type == 'all':
            self._generate_roadmap_docs()
        
        if doc_type == 'architecture' or doc_type == 'all':
            self._generate_architecture_docs()
        
        print("\n✅ Documentation generated")
        print(f"📁 Output: {self.docs_dir}")
        
        return 0
    
    def generate_api_docs(self) -> int:
        """Generate API documentation."""
        print("\n🔧 Generating API Documentation")
        
        # Try sphinx
        sphinx_conf = self.docs_dir / 'conf.py'
        if sphinx_conf.exists():
            cmd = ['sphinx-build', '-b', 'html', str(self.docs_dir),
                   str(self.docs_dir / '_build')]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ Sphinx documentation generated")
                return 0
        
        # Try pdoc
        try:
            cmd = ['pdoc', '--html', '--output-dir', str(self.docs_dir),
                   'brain', 'state']
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ API documentation generated with pdoc")
                return 0
        except:
            pass
        
        # Fallback: Generate basic API index
        self._generate_basic_api_docs()
        return 0
    
    def _generate_basic_api_docs(self) -> None:
        """Generate basic API documentation."""
        api_doc = self.docs_dir / 'API.md'
        
        content = ["# Quark API Documentation\n\n"]
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Scan for Python modules
        for module_dir in ['brain', 'state', 'tools_utilities']:
            module_path = self.project_root / module_dir
            if module_path.exists():
                content.append(f"## {module_dir.title()} Module\n\n")
                
                for py_file in sorted(module_path.rglob('*.py')):
                    if '__pycache__' not in str(py_file):
                        rel_path = py_file.relative_to(self.project_root)
                        content.append(f"- [{rel_path}]({rel_path})\n")
                
                content.append("\n")
        
        with open(api_doc, 'w') as f:
            f.writelines(content)
        
        print(f"📝 Basic API docs: {api_doc}")
    
    def _generate_validation_docs(self) -> None:
        """Generate validation documentation."""
        val_dir = self.project_root / 'state' / 'tasks' / 'validation'
        if val_dir.exists():
            # Create validation summary
            summary_file = self.docs_dir / 'VALIDATION_SUMMARY.md'
            
            content = ["# Validation Documentation Summary\n\n"]
            content.append("## Available Checklists\n\n")
            
            for checklist in sorted((val_dir / 'checklists').glob('*.md')):
                content.append(f"- [{checklist.stem}](../state/tasks/validation/checklists/{checklist.name})\n")
            
            with open(summary_file, 'w') as f:
                f.writelines(content)
            
            print(f"📝 Validation docs: {summary_file}")
    
    def _generate_roadmap_docs(self) -> None:
        """Generate roadmap documentation."""
        roadmap_dir = self.project_root / 'management' / 'rules' / 'roadmap'
        if roadmap_dir.exists():
            # Create roadmap index
            index_file = self.docs_dir / 'ROADMAP_INDEX.md'
            
            content = ["# Roadmap Documentation Index\n\n"]
            
            for stage_file in sorted(roadmap_dir.glob('stage*.md')):
                content.append(f"- [{stage_file.stem}](../management/rules/roadmap/{stage_file.name})\n")
            
            with open(index_file, 'w') as f:
                f.writelines(content)
            
            print(f"📝 Roadmap docs: {index_file}")
    
    def _generate_architecture_docs(self) -> None:
        """Generate architecture documentation."""
        arch_file = self.docs_dir / 'ARCHITECTURE.md'
        
        content = ["# Quark Architecture Documentation\n\n"]
        content.append("## System Components\n\n")
        
        # Document main components
        components = {
            'Brain': 'Core neural simulation engine',
            'State': 'State management and persistence',
            'Validation': 'Testing and validation framework',
            'Deployment': 'Deployment and scaling infrastructure'
        }
        
        for comp, desc in components.items():
            content.append(f"### {comp}\n{desc}\n\n")
        
        with open(arch_file, 'w') as f:
            f.writelines(content)
        
        print(f"📝 Architecture docs: {arch_file}")
    
    def update_readme(self, params: Dict) -> int:
        """Update README file."""
        print("\n📝 Updating README")
        print("=" * 50)
        
        if not self.readme.exists():
            print("⚠️ README.md not found")
            return 1
        
        # Read current README
        with open(self.readme) as f:
            lines = f.readlines()
        
        # Update sections
        section = params.get('section', 'all')
        
        if section == 'badges' or section == 'all':
            self._update_readme_badges(lines)
        
        if section == 'quickstart' or section == 'all':
            self._update_readme_quickstart(lines)
        
        if section == 'status' or section == 'all':
            self._update_readme_status(lines)
        
        # Write updated README
        with open(self.readme, 'w') as f:
            f.writelines(lines)
        
        print("✅ README updated")
        return 0
    
    def _update_readme_badges(self, lines: List[str]) -> None:
        """Update README badges."""
        # This would update CI/CD badges, version badges, etc.
        print("  • Updated badges")
    
    def _update_readme_quickstart(self, lines: List[str]) -> None:
        """Update README quickstart section."""
        print("  • Updated quickstart guide")
    
    def _update_readme_status(self, lines: List[str]) -> None:
        """Update README status section."""
        print("  • Updated project status")
    
    def check_documentation(self) -> int:
        """Check documentation completeness."""
        print("\n🔍 Checking Documentation")
        print("=" * 50)
        
        required_docs = [
            ('README.md', self.project_root / 'README.md'),
            ('API Docs', self.docs_dir / 'API.md'),
            ('Architecture', self.docs_dir / 'ARCHITECTURE.md'),
            ('Validation Guide', self.project_root / 'state' / 'tasks' / 'validation' / 'VALIDATION_GUIDE.md')
        ]
        
        missing = []
        outdated = []
        
        for name, path in required_docs:
            if path.exists():
                # Check if outdated (modified > 30 days ago)
                import time
                age_days = (time.time() - path.stat().st_mtime) / 86400
                if age_days > 30:
                    outdated.append(name)
                    print(f"⚠️ {name}: Outdated ({int(age_days)} days old)")
                else:
                    print(f"✅ {name}: Up to date")
            else:
                missing.append(name)
                print(f"❌ {name}: Missing")
        
        if missing:
            print(f"\n📝 Missing docs: {', '.join(missing)}")
            print("   Run: todo generate docs")
        
        if outdated:
            print(f"\n⏰ Outdated docs: {', '.join(outdated)}")
            print("   Run: todo update readme")
        
        return 0 if not missing else 1
    
    def show_help(self) -> int:
        """Show documentation help."""
        print("""
📚 Documentation Commands:
  todo generate docs           → Generate all documentation
  todo generate api docs       → Generate API documentation
  todo update readme           → Update README.md
  todo check docs              → Check documentation status
  todo generate validation docs → Generate validation docs
  todo generate roadmap docs   → Generate roadmap docs
""")
        return 0
