#!/usr/bin/env python3
"""
Data Downloader for Brainstem Segmentation Datasets

Fetches actual data from Allen Brain Atlas and other sources.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllenBrainDownloader:
    """Downloads data from Allen Brain Atlas Developing Mouse API."""
    
    BASE_URL = "http://api.brain-map.org/api/v2"
    
    def __init__(self, data_dir: Path):
        """Initialize downloader with storage directory.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw" / "allen"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def get_developing_mouse_experiments(self) -> List[Dict]:
        """Fetch list of developing mouse brain experiments.
        
        Returns:
            List of experiment metadata dictionaries
        """
        query_url = f"{self.BASE_URL}/data/query.json"
        params = {
            "criteria": "model::SectionDataSet,rma::criteria,[failed$eq'false'],products[abbreviation$eq'DevMouse']",
            "include": "specimen(donor(age)),plane_of_section",
            "num_rows": 50,
            "start_row": 0
        }
        
        url = f"{query_url}?{urllib.parse.urlencode(params)}"
        logger.info(f"Fetching experiments from: {url}")
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read())
                experiments = data.get("msg", [])
                logger.info(f"Found {len(experiments)} developing mouse experiments")
                return experiments
        except Exception as e:
            logger.error(f"Failed to fetch experiments: {e}")
            return []
    
    def download_experiment_metadata(self, experiments: List[Dict]) -> None:
        """Save experiment metadata to JSON files.
        
        Args:
            experiments: List of experiment dictionaries
        """
        metadata_dir = self.raw_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Filter for brainstem-relevant stages
        target_ages = ["E11.5", "E13.5", "E15.5", "E18.5"]
        filtered_exps = []
        
        for exp in experiments:
            age = exp.get("specimen", {}).get("donor", {}).get("age", {}).get("name", "")
            if any(target in age for target in target_ages):
                filtered_exps.append(exp)
                
                # Save individual experiment metadata
                exp_file = metadata_dir / f"experiment_{exp['id']}.json"
                with open(exp_file, 'w') as f:
                    json.dump(exp, f, indent=2)
        
        logger.info(f"Saved metadata for {len(filtered_exps)} relevant experiments")
        
        # Save summary
        summary_file = metadata_dir / "experiments_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "total_experiments": len(filtered_exps),
                "ages": target_ages,
                "experiments": filtered_exps
            }, f, indent=2)
    
    def get_section_images_urls(self, experiment_id: int, limit: int = 10) -> List[str]:
        """Get download URLs for section images from an experiment.
        
        Args:
            experiment_id: Allen experiment ID
            limit: Maximum number of sections to retrieve
            
        Returns:
            List of image download URLs
        """
        query_url = f"{self.BASE_URL}/data/query.json"
        params = {
            "criteria": f"model::SectionImage,rma::criteria,[data_set_id$eq{experiment_id}]",
            "num_rows": limit,
            "start_row": 0
        }
        
        url = f"{query_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read())
                images = data.get("msg", [])
                
                # Build download URLs
                urls = []
                for img in images:
                    download_url = f"http://api.brain-map.org/api/v2/section_image_download/{img['id']}"
                    urls.append(download_url)
                
                return urls
        except Exception as e:
            logger.error(f"Failed to get image URLs for experiment {experiment_id}: {e}")
            return []


class DevCCFDownloader:
    """Downloads DevCCF reference atlases."""
    
    # Direct download URLs for DevCCF data
    DEVCCF_URLS = {
        "E11.5": "https://download.brainimagelibrary.org/02/0a/020abf5af3e68902/E11.5_DevCCF_Annotations.nii.gz",
        "E13.5": "https://download.brainimagelibrary.org/02/0a/020abf5af3e68902/E13.5_DevCCF_Annotations.nii.gz", 
        "E15.5": "https://download.brainimagelibrary.org/02/0a/020abf5af3e68902/E15.5_DevCCF_Annotations.nii.gz"
    }
    
    def __init__(self, data_dir: Path):
        """Initialize downloader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.template_dir = data_dir / "templates" / "devccf"
        self.template_dir.mkdir(parents=True, exist_ok=True)
    
    def download_atlas(self, stage: str) -> bool:
        """Download a specific DevCCF atlas.
        
        Args:
            stage: Developmental stage (e.g., "E11.5")
            
        Returns:
            True if successful, False otherwise
        """
        if stage not in self.DEVCCF_URLS:
            logger.error(f"Unknown stage: {stage}")
            return False
        
        url = self.DEVCCF_URLS[stage]
        output_file = self.template_dir / f"DevCCF_{stage}_Annotations.nii.gz"
        
        if output_file.exists():
            logger.info(f"Atlas for {stage} already exists at {output_file}")
            return True
        
        logger.info(f"Downloading DevCCF {stage} from {url}")
        logger.info("This may take several minutes...")
        
        try:
            urllib.request.urlretrieve(url, output_file)
            logger.info(f"Successfully downloaded {stage} atlas to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {stage} atlas: {e}")
            return False
    
    def download_all(self) -> Dict[str, bool]:
        """Download all DevCCF atlases.
        
        Returns:
            Dictionary mapping stage to download success status
        """
        results = {}
        for stage in self.DEVCCF_URLS.keys():
            results[stage] = self.download_atlas(stage)
            time.sleep(1)  # Be polite to the server
        return results


def collect_priority_data(data_dir: Optional[Path] = None) -> None:
    """Main function to collect priority datasets.
    
    Args:
        data_dir: Base directory for data storage
    """
    if data_dir is None:
        data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    
    print("\n🔄 Starting data collection...")
    
    # Download Allen Brain Atlas data
    print("\n📊 Fetching Allen Brain Atlas data...")
    allen_downloader = AllenBrainDownloader(data_dir)
    experiments = allen_downloader.get_developing_mouse_experiments()
    
    if experiments:
        allen_downloader.download_experiment_metadata(experiments)
        
        # Download sample images from first experiment
        if experiments:
            exp_id = experiments[0]["id"]
            print(f"\n🖼️ Getting sample images from experiment {exp_id}...")
            image_urls = allen_downloader.get_section_images_urls(exp_id, limit=5)
            
            # Save URLs for manual download
            urls_file = data_dir / "raw" / "allen" / "sample_image_urls.txt"
            with open(urls_file, 'w') as f:
                for url in image_urls:
                    f.write(f"{url}\n")
            print(f"Saved {len(image_urls)} image URLs to {urls_file}")
    
    # Download DevCCF atlases
    print("\n🧠 Downloading DevCCF reference atlases...")
    devccf_downloader = DevCCFDownloader(data_dir)
    results = devccf_downloader.download_all()
    
    # Summary
    print("\n📈 Data Collection Summary:")
    print(f"- Allen experiments found: {len(experiments)}")
    print(f"- DevCCF atlases downloaded: {sum(results.values())}/{len(results)}")
    
    successful = [stage for stage, success in results.items() if success]
    if successful:
        print(f"  ✅ Successful: {', '.join(successful)}")
    
    failed = [stage for stage, success in results.items() if not success]
    if failed:
        print(f"  ❌ Failed: {', '.join(failed)}")
    
    print("\n✅ Data collection complete!")
    print(f"📁 Data stored in: {data_dir}")


if __name__ == "__main__":
    collect_priority_data()
