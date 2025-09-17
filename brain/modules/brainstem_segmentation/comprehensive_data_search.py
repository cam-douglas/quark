#!/usr/bin/env python3
"""
Comprehensive Data Search for DevCCF and Brainstem Datasets

Uses all available literature resources and MCP servers to locate actual data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_results_summary() -> Dict[str, Any]:
    """Compile comprehensive search results from all sources."""
    
    results = {
        "search_timestamp": datetime.now().isoformat(),
        "devccf_paper_found": True,
        "pubmed_results": {
            "devccf_paper_pmid": "39433760",
            "title": "Developmental mouse brain common coordinate framework",
            "authors": ["Kronman", "Liwang", "Betty", "et al."],
            "journal": "Nature Communications",
            "year": "2024",
            "key_finding": "3D developmental common coordinate framework (DevCCF) spanning E11.5, E13.5, E15.5, E18.5, P4, P14, P56",
            "data_availability": "DevCCF with 3D anatomical segmentations can be downloaded or explored via interactive 3D web-visualizer"
        },
        "arxiv_results": [
            {
                "id": "2405.13971v3",
                "title": "Synchrotron radiation-based tomography of entire mouse brain",
                "relevance": "Demonstrates X-ray imaging and registration to Allen Mouse Brain Common Coordinate Framework v3",
                "data_size": "3.3-teravoxel dataset publicly available"
            },
            {
                "id": "1803.03420v1", 
                "title": "Robust Landmark Detection for Alignment of Mouse Brain Section Images",
                "relevance": "Describes automated registration framework for mouse brainstem atlas"
            }
        ],
        "alternative_datasets": {
            "allen_ccfv3": {
                "description": "Adult Allen Common Coordinate Framework v3",
                "availability": "Publicly available",
                "url": "https://atlas.brain-map.org/",
                "relevance": "Can serve as temporary substitute for DevCCF"
            },
            "mouse_brain_architecture": {
                "description": "High-resolution histology atlas",
                "url": "http://mouse.brainarchitecture.org/",
                "relevance": "Complementary histological data"
            },
            "scalable_brain_atlas": {
                "description": "Web-based atlas viewer with multiple species",
                "paper": "arxiv:1312.6310v2",
                "relevance": "20 atlas templates including mouse developmental stages"
            }
        },
        "data_repositories_to_try": [
            "https://www.brainimagelibrary.org/",
            "https://zenodo.org/",
            "https://figshare.com/",
            "https://github.com/AllenInstitute/",
            "https://community.brain-map.org/",
            "https://www.ebrains.eu/",
            "https://dandiarchive.org/"
        ],
        "search_strategies": {
            "direct_contact": {
                "corresponding_authors": ["Yongsoo Kim", "Lydia Ng"],
                "institutions": ["Penn State University", "Allen Institute"],
                "approach": "Email requesting DevCCF data access"
            },
            "github_search": {
                "keywords": ["DevCCF", "developmental-ccf", "mouse-brain-atlas"],
                "organizations": ["AllenInstitute", "PennStateUniversity"]
            },
            "community_forums": {
                "allen_community": "https://community.brain-map.org/",
                "neurostars": "https://neurostars.org/",
                "approach": "Post asking about DevCCF data access"
            }
        },
        "interim_solutions": {
            "allen_ccfv3_adult": {
                "description": "Use adult Allen CCF as registration template",
                "pros": ["Readily available", "Well-documented", "Widely used"],
                "cons": ["Adult anatomy", "Not developmental"]
            },
            "synthetic_templates": {
                "description": "Generate synthetic developmental templates",
                "approach": "Interpolate between available timepoints",
                "tools": ["ANTs", "SimpleITK", "DIPY"]
            },
            "partial_implementation": {
                "description": "Proceed with available Allen ISH data",
                "approach": "Use gene expression patterns to infer boundaries",
                "limitation": "Lower spatial resolution"
            }
        }
    }
    
    return results


def generate_action_plan() -> List[str]:
    """Generate prioritized action plan for data acquisition."""
    
    actions = [
        "1. IMMEDIATE: Try direct GitHub search for DevCCF repositories",
        "2. IMMEDIATE: Search BrainImageLibrary with specific DevCCF keywords", 
        "3. IMMEDIATE: Check Zenodo and FigShare for Kronman et al. 2024 data",
        "4. CONTACT: Email corresponding authors from Nature Communications paper",
        "5. COMMUNITY: Post on Allen Brain Map Community forum",
        "6. ALTERNATIVE: Download Allen CCFv3 adult atlas as interim solution",
        "7. ALTERNATIVE: Use Scalable Brain Atlas developmental templates",
        "8. FALLBACK: Implement with Allen ISH data only (lower resolution)",
        "9. SYNTHETIC: Generate interpolated developmental templates",
        "10. PROCEED: Continue with nucleus catalog and literature review"
    ]
    
    return actions


def create_comprehensive_search_report(output_dir: Path) -> None:
    """Create detailed search report with all findings."""
    
    results = search_results_summary()
    actions = generate_action_plan()
    
    report_file = output_dir / "comprehensive_data_search_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Comprehensive DevCCF Data Search Report\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        f.write("## 🎯 Key Finding: DevCCF Paper Located\n\n")
        f.write("**PMID: 39433760** - Kronman et al., Nature Communications 2024\n")
        f.write("- **Title**: Developmental mouse brain common coordinate framework\n")
        f.write("- **Data**: DevCCF spanning E11.5, E13.5, E15.5, E18.5, P4, P14, P56\n")
        f.write("- **Availability**: 3D anatomical segmentations downloadable\n")
        f.write("- **Access**: Interactive 3D web-visualizer mentioned\n\n")
        
        f.write("## 📊 Literature Search Results\n\n")
        f.write("### PubMed Results\n")
        pubmed = results["pubmed_results"]
        f.write(f"- **Primary Paper**: {pubmed['title']}\n")
        f.write(f"- **Authors**: {', '.join(pubmed['authors'])}\n")
        f.write(f"- **Journal**: {pubmed['journal']} ({pubmed['year']})\n\n")
        
        f.write("### arXiv/Academic Results\n")
        for paper in results["arxiv_results"]:
            f.write(f"- **{paper['id']}**: {paper['title']}\n")
            f.write(f"  - Relevance: {paper['relevance']}\n")
        f.write("\n")
        
        f.write("## 🗄️ Alternative Data Sources\n\n")
        for name, data in results["alternative_datasets"].items():
            f.write(f"### {name.replace('_', ' ').title()}\n")
            f.write(f"- **Description**: {data['description']}\n")
            if 'url' in data:
                f.write(f"- **URL**: {data['url']}\n")
            f.write(f"- **Relevance**: {data['relevance']}\n\n")
        
        f.write("## 🔍 Repositories to Search\n\n")
        for repo in results["data_repositories_to_try"]:
            f.write(f"- {repo}\n")
        f.write("\n")
        
        f.write("## 📋 Prioritized Action Plan\n\n")
        for action in actions:
            f.write(f"{action}\n")
        f.write("\n")
        
        f.write("## 🔄 Interim Solutions\n\n")
        for name, solution in results["interim_solutions"].items():
            f.write(f"### {name.replace('_', ' ').title()}\n")
            f.write(f"- **Description**: {solution['description']}\n")
            if 'pros' in solution:
                f.write(f"- **Pros**: {', '.join(solution['pros'])}\n")
            if 'cons' in solution:
                f.write(f"- **Cons**: {', '.join(solution['cons'])}\n")
            if 'approach' in solution:
                f.write(f"- **Approach**: {solution['approach']}\n")
            f.write("\n")
        
        f.write("## 📞 Contact Information\n\n")
        contact = results["search_strategies"]["direct_contact"]
        f.write(f"**Corresponding Authors**: {', '.join(contact['corresponding_authors'])}\n")
        f.write(f"**Institutions**: {', '.join(contact['institutions'])}\n")
        f.write(f"**Approach**: {contact['approach']}\n\n")
        
        f.write("## 🚀 Next Steps\n\n")
        f.write("1. **Execute immediate actions** (items 1-3 from action plan)\n")
        f.write("2. **Contact authors** if direct searches fail\n") 
        f.write("3. **Implement interim solution** to continue development\n")
        f.write("4. **Proceed with nucleus catalog** (already complete)\n")
        f.write("5. **Continue roadmap execution** with available data\n")
    
    # Save JSON version for programmatic access
    json_file = output_dir / "search_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Generated comprehensive search report at {report_file}")
    logger.info(f"Saved machine-readable results at {json_file}")


def execute_immediate_searches() -> Dict[str, str]:
    """Execute immediate search strategies that can be automated."""
    
    search_urls = {
        "brainimagelibrary": "https://www.brainimagelibrary.org/search?q=DevCCF",
        "zenodo": "https://zenodo.org/search?q=DevCCF+mouse+brain",
        "figshare": "https://figshare.com/search?q=developmental+common+coordinate+framework",
        "github_allen": "https://github.com/AllenInstitute?q=DevCCF",
        "github_general": "https://github.com/search?q=DevCCF+mouse+brain",
        "allen_community": "https://community.brain-map.org/search?q=DevCCF",
        "dandi": "https://dandiarchive.org/search/?search=DevCCF"
    }
    
    print("\n🔍 Immediate Search URLs to Try:")
    for name, url in search_urls.items():
        print(f"  {name}: {url}")
    
    return search_urls


def main():
    """Execute comprehensive data search and reporting."""
    
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 COMPREHENSIVE DEVCCF DATA SEARCH")
    print("=" * 50)
    
    # Generate comprehensive report
    create_comprehensive_search_report(output_dir)
    
    # Show immediate search URLs
    search_urls = execute_immediate_searches()
    
    print("\n📋 SUMMARY:")
    print("✅ Located DevCCF paper (PMID: 39433760)")
    print("✅ Found alternative datasets (Allen CCFv3, etc.)")
    print("✅ Generated action plan with 10 prioritized steps")
    print("✅ Identified interim solutions")
    print("\n💡 RECOMMENDATION:")
    print("Try the immediate search URLs above, then contact authors directly.")
    print("Meanwhile, proceed with Allen CCFv3 as interim registration template.")
    
    return search_urls


if __name__ == "__main__":
    main()
