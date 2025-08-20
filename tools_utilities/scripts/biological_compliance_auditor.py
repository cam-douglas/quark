#!/usr/bin/env python3
"""
🧬 Biological Compliance Auditor
Comprehensive marker distribution analysis and biological fidelity validation
"""

import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

class BiologicalComplianceAuditor:
    """Comprehensive biological compliance auditing system"""
    
    def __init__(self):
        self.memory_dir = Path("memory")
        self.load_semantic_network()
        self.define_biological_standards()
        
    def load_semantic_network(self):
        """Load the semantic network and metadata"""
        try:
            with open(self.memory_dir / "metadata.json") as f:
                self.metadata = json.load(f)
            
            with open(self.memory_dir / "rule_graph.json") as f:
                graph_data = json.load(f)
                self.graph = nx.node_link_graph(graph_data)
            
            with open(self.memory_dir / "summaries.json") as f:
                self.summaries = json.load(f)
                
            print(f"✅ Loaded semantic network: {len(self.metadata)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to load semantic network: {e}")
            raise
    
    def define_biological_standards(self):
        """Define biological standards and expected marker distributions"""
        self.biological_standards = {
            "critical_markers": {
                "GFAP": {"min_percentage": 15.0, "max_percentage": 25.0, "function": "Structural enforcement"},
                "NeuN": {"min_percentage": 20.0, "max_percentage": 30.0, "function": "Core identity"}
            },
            "essential_markers": {
                "GAP43": {"min_percentage": 20.0, "max_percentage": 35.0, "function": "Learning/plasticity"},
                "NSE": {"min_percentage": 10.0, "max_percentage": 20.0, "function": "Neural development"},
                "S100B": {"min_percentage": 10.0, "max_percentage": 20.0, "function": "Support/regulation"}
            },
            "specialized_markers": {
                "Tau": {"min_percentage": 5.0, "max_percentage": 15.0, "function": "Cognitive stability"},
                "MBP": {"min_percentage": 3.0, "max_percentage": 10.0, "function": "Insulation/protection"},
                "Vimentin": {"min_percentage": 15.0, "max_percentage": 30.0, "function": "Flexibility"}
            },
            "marker_relationships": {
                "GFAP": ["Vimentin", "S100B"],  # GFAP co-expresses with these
                "NeuN": ["GAP43", "NSE"],       # NeuN co-expresses with these
                "GAP43": ["NSE", "Tau"],        # GAP43 co-expresses with these
                "S100B": ["GFAP", "Vimentin"]   # S100B co-expresses with these
            }
        }
    
    def comprehensive_audit(self) -> Dict[str, Any]:
        """Perform comprehensive biological compliance audit"""
        print("🧬 Performing comprehensive biological compliance audit...")
        
        audit_results = {
            "marker_distribution": self._analyze_marker_distribution(),
            "marker_relationships": self._analyze_marker_relationships(),
            "priority_compliance": self._analyze_priority_compliance(),
            "biological_fidelity": self._analyze_biological_fidelity(),
            "compliance_score": 0.0,
            "recommendations": []
        }
        
        # Calculate overall compliance score
        audit_results["compliance_score"] = self._calculate_overall_compliance(audit_results)
        
        # Generate recommendations
        audit_results["recommendations"] = self._generate_recommendations(audit_results)
        
        return audit_results
    
    def _analyze_marker_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of biological markers across chunks"""
        print("   📊 Analyzing marker distribution...")
        
        # Count markers across all chunks
        marker_counts = Counter()
        file_marker_distribution = defaultdict(Counter)
        
        for chunk in self.metadata:
            markers = chunk.get("markers", [])
            file_name = chunk.get("file", "unknown")
            
            for marker in markers:
                marker_counts[marker] += 1
                file_marker_distribution[file_name][marker] += 1
        
        # Calculate percentages
        total_chunks = len(self.metadata)
        marker_percentages = {marker: (count / total_chunks) * 100 for marker, count in marker_counts.items()}
        
        # Check compliance with standards
        compliance_checks = {}
        for marker, percentage in marker_percentages.items():
            if marker in self.biological_standards["critical_markers"]:
                standard = self.biological_standards["critical_markers"][marker]
                compliance_checks[marker] = {
                    "percentage": percentage,
                    "standard": standard,
                    "compliant": standard["min_percentage"] <= percentage <= standard["max_percentage"],
                    "severity": "critical"
                }
            elif marker in self.biological_standards["essential_markers"]:
                standard = self.biological_standards["essential_markers"][marker]
                compliance_checks[marker] = {
                    "percentage": percentage,
                    "standard": standard,
                    "compliant": standard["min_percentage"] <= percentage <= standard["max_percentage"],
                    "severity": "essential"
                }
            elif marker in self.biological_standards["specialized_markers"]:
                standard = self.biological_standards["specialized_markers"][marker]
                compliance_checks[marker] = {
                    "percentage": percentage,
                    "standard": standard,
                    "compliant": standard["min_percentage"] <= percentage <= standard["max_percentage"],
                    "severity": "specialized"
                }
        
        return {
            "total_chunks": total_chunks,
            "marker_counts": dict(marker_counts),
            "marker_percentages": marker_percentages,
            "compliance_checks": compliance_checks,
            "file_distribution": dict(file_marker_distribution)
        }
    
    def _analyze_marker_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between biological markers"""
        print("   🔗 Analyzing marker relationships...")
        
        relationship_analysis = {}
        
        for marker, expected_partners in self.biological_standards["marker_relationships"].items():
            # Find chunks with this marker
            marker_chunks = [chunk for chunk in self.metadata if marker in chunk.get("markers", [])]
            
            if not marker_chunks:
                continue
            
            # Analyze co-expression patterns
            co_expression = defaultdict(int)
            total_marker_chunks = len(marker_chunks)
            
            for chunk in marker_chunks:
                chunk_markers = chunk.get("markers", [])
                for other_marker in chunk_markers:
                    if other_marker != marker:
                        co_expression[other_marker] += 1
            
            # Calculate co-expression percentages
            co_expression_percentages = {
                other_marker: (count / total_marker_chunks) * 100 
                for other_marker, count in co_expression.items()
            }
            
            # Check expected relationships
            relationship_quality = {}
            for expected_partner in expected_partners:
                if expected_partner in co_expression_percentages:
                    percentage = co_expression_percentages[expected_partner]
                    relationship_quality[expected_partner] = {
                        "observed_percentage": percentage,
                        "expected": True,
                        "strength": "strong" if percentage > 30 else "moderate" if percentage > 15 else "weak"
                    }
                else:
                    relationship_quality[expected_partner] = {
                        "observed_percentage": 0.0,
                        "expected": True,
                        "strength": "missing"
                    }
            
            relationship_analysis[marker] = {
                "total_chunks": total_marker_chunks,
                "co_expression_patterns": co_expression_percentages,
                "expected_relationships": relationship_quality
            }
        
        return relationship_analysis
    
    def _analyze_priority_compliance(self) -> Dict[str, Any]:
        """Analyze compliance with priority-based biological standards"""
        print("   🎯 Analyzing priority compliance...")
        
        priority_analysis = defaultdict(lambda: {"chunks": [], "markers": Counter(), "compliance": {}})
        
        for chunk in self.metadata:
            filename = chunk.get("file", "")
            # Extract priority from filename (e.g., "01-index.md" -> priority 1)
            import re
            match = re.match(r"(\d+)-", filename)
            if match:
                priority = int(match.group(1))
                priority_analysis[priority]["chunks"].append(chunk)
                
                # Count markers for this priority level
                markers = chunk.get("markers", [])
                for marker in markers:
                    priority_analysis[priority]["markers"][marker] += 1
        
        # Analyze each priority level
        for priority, data in priority_analysis.items():
            total_chunks = len(data["chunks"])
            if total_chunks == 0:
                continue
            
            # Calculate marker percentages for this priority
            marker_percentages = {
                marker: (count / total_chunks) * 100 
                for marker, count in data["markers"].items()
            }
            
            # Check compliance based on priority level
            compliance_checks = {}
            if priority <= 2:  # High priority - should have critical markers
                for marker in ["GFAP", "NeuN"]:
                    percentage = marker_percentages.get(marker, 0.0)
                    compliance_checks[marker] = {
                        "required": True,
                        "observed_percentage": percentage,
                        "compliant": percentage >= 20.0
                    }
            
            data["compliance"] = compliance_checks
        
        return dict(priority_analysis)
    
    def _analyze_biological_fidelity(self) -> Dict[str, Any]:
        """Analyze overall biological fidelity of the system"""
        print("   🧬 Analyzing biological fidelity...")
        
        fidelity_metrics = {
            "marker_diversity": len(set(marker for chunk in self.metadata for marker in chunk.get("markers", []))),
            "critical_marker_coverage": 0.0,
            "relationship_integrity": 0.0,
            "priority_biological_alignment": 0.0
        }
        
        # Calculate critical marker coverage
        critical_markers = {"GFAP", "NeuN"}
        chunks_with_critical = sum(
            1 for chunk in self.metadata 
            if any(marker in critical_markers for marker in chunk.get("markers", []))
        )
        fidelity_metrics["critical_marker_coverage"] = (chunks_with_critical / len(self.metadata)) * 100
        
        # Calculate relationship integrity
        total_expected_relationships = sum(len(partners) for partners in self.biological_standards["marker_relationships"].values())
        observed_relationships = 0
        
        for marker, expected_partners in self.biological_standards["marker_relationships"].items():
            marker_chunks = [chunk for chunk in self.metadata if marker in chunk.get("markers", [])]
            for chunk in marker_chunks:
                chunk_markers = chunk.get("markers", [])
                for partner in expected_partners:
                    if partner in chunk_markers:
                        observed_relationships += 1
        
        if total_expected_relationships > 0:
            fidelity_metrics["relationship_integrity"] = (observed_relationships / total_expected_relationships) * 100
        
        return fidelity_metrics
    
    def _calculate_overall_compliance(self, audit_results: Dict[str, Any]) -> float:
        """Calculate overall biological compliance score"""
        scores = []
        
        # Marker distribution compliance
        distribution = audit_results["marker_distribution"]
        compliant_checks = sum(1 for check in distribution["compliance_checks"].values() if check["compliant"])
        total_checks = len(distribution["compliance_checks"])
        if total_checks > 0:
            scores.append(compliant_checks / total_checks)
        
        # Priority compliance
        priority = audit_results["priority_compliance"]
        priority_scores = []
        for data in priority.values():
            if data["compliance"]:
                compliant = sum(1 for check in data["compliance"].values() if check.get("compliant", False))
                total = len(data["compliance"])
                if total > 0:
                    priority_scores.append(compliant / total)
        
        if priority_scores:
            scores.append(sum(priority_scores) / len(priority_scores))
        
        # Biological fidelity
        fidelity = audit_results["biological_fidelity"]
        if fidelity["critical_marker_coverage"] > 0:
            scores.append(min(fidelity["critical_marker_coverage"] / 100, 1.0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving biological compliance"""
        recommendations = []
        
        # Marker distribution recommendations
        distribution = audit_results["marker_distribution"]
        for marker, check in distribution["compliance_checks"].items():
            if not check["compliant"]:
                percentage = check["percentage"]
                standard = check["standard"]
                if percentage < standard["min_percentage"]:
                    recommendations.append(
                        f"Increase {marker} marker presence: currently {percentage:.1f}%, "
                        f"should be {standard['min_percentage']:.1f}%-{standard['max_percentage']:.1f}%"
                    )
                elif percentage > standard["max_percentage"]:
                    recommendations.append(
                        f"Reduce {marker} marker presence: currently {percentage:.1f}%, "
                        f"should be {standard['min_percentage']:.1f}%-{standard['max_percentage']:.1f}%"
                    )
        
        # Priority compliance recommendations
        priority = audit_results["priority_compliance"]
        for priority_level, data in priority.items():
            if priority_level <= 2 and data["compliance"]:  # High priority levels
                for marker, check in data["compliance"].items():
                    if check["required"] and not check["compliant"]:
                        recommendations.append(
                            f"Priority {priority_level}: Ensure {marker} marker presence "
                            f"(currently {check['observed_percentage']:.1f}%, should be ≥20%)"
                        )
        
        # Biological fidelity recommendations
        fidelity = audit_results["biological_fidelity"]
        if fidelity["critical_marker_coverage"] < 50:
            recommendations.append(
                f"Increase critical marker coverage: currently {fidelity['critical_marker_coverage']:.1f}%, "
                f"should be ≥50%"
            )
        
        if fidelity["relationship_integrity"] < 70:
            recommendations.append(
                f"Improve marker relationship integrity: currently {fidelity['relationship_integrity']:.1f}%, "
                f"should be ≥70%"
            )
        
        return recommendations
    
    def generate_audit_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate a comprehensive audit report"""
        report = []
        report.append("🧬 BIOLOGICAL COMPLIANCE AUDIT REPORT")
        report.append("=" * 60)
        report.append(f"Overall Compliance Score: {audit_results['compliance_score']:.1%}")
        report.append("")
        
        # Marker distribution summary
        report.append("📊 MARKER DISTRIBUTION ANALYSIS")
        report.append("-" * 40)
        distribution = audit_results["marker_distribution"]
        for marker, check in distribution["compliance_checks"].items():
            status = "✅" if check["compliant"] else "❌"
            report.append(f"{status} {marker}: {check['percentage']:.1f}% "
                         f"({check['standard']['min_percentage']:.1f}%-{check['standard']['max_percentage']:.1f}%)")
        report.append("")
        
        # Priority compliance summary
        report.append("🎯 PRIORITY COMPLIANCE ANALYSIS")
        report.append("-" * 40)
        priority = audit_results["priority_compliance"]
        for priority_level in sorted(priority.keys()):
            data = priority[priority_level]
            if data["compliance"]:
                compliant = sum(1 for check in data["compliance"].values() if check.get("compliant", False))
                total = len(data["compliance"])
                percentage = (compliant / total) * 100 if total > 0 else 0
                report.append(f"Priority {priority_level}: {percentage:.1f}% compliant ({compliant}/{total})")
        report.append("")
        
        # Biological fidelity summary
        report.append("🧬 BIOLOGICAL FIDELITY METRICS")
        report.append("-" * 40)
        fidelity = audit_results["biological_fidelity"]
        report.append(f"Marker Diversity: {fidelity['marker_diversity']} unique markers")
        report.append(f"Critical Marker Coverage: {fidelity['critical_marker_coverage']:.1f}%")
        report.append(f"Relationship Integrity: {fidelity['relationship_integrity']:.1f}%")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        for i, recommendation in enumerate(audit_results["recommendations"], 1):
            report.append(f"{i}. {recommendation}")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    print("🧬 Biological Compliance Auditor")
    print("=" * 50)
    
    try:
        auditor = BiologicalComplianceAuditor()
        
        # Perform comprehensive audit
        audit_results = auditor.comprehensive_audit()
        
        # Generate and display report
        report = auditor.generate_audit_report(audit_results)
        print("\n" + report)
        
        print(f"\n✅ Audit completed successfully!")
        print(f"   Overall compliance: {audit_results['compliance_score']:.1%}")
        print(f"   Recommendations: {len(audit_results['recommendations'])}")
        
    except Exception as e:
        print(f"❌ Audit failed: {e}")

if __name__ == "__main__":
    main()
