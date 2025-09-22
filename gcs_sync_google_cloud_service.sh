#!/bin/bash
"""
GCS Sync to Google Cloud Service - Wrapper script with activation words

ACTIVATION WORDS: 
  - sync to gcs
  - sync to google cloud  
  - sync to google cloud service
  - download from gcs
  - download from google cloud
  - download from google cloud service

Usage:
  ./gcs_sync_google_cloud_service.sh                    # Sync local data/ to GCS buckets
  ./gcs_sync_google_cloud_service.sh --preview          # Show what would be synced
  ./gcs_sync_google_cloud_service.sh --download         # Download from GCS to local
  ./gcs_sync_google_cloud_service.sh --dry-run          # Test run without actual sync
  ./gcs_sync_google_cloud_service.sh --no-cleanup       # Skip automatic cleanup
"""

# Change to script directory
cd "$(dirname "$0")"

# Run the Python sync script
python3 tools_utilities/gcs_sync_to_google_cloud.py "$@"
