#!/bin/bash
#
# GCS Sync to Google Cloud Service - Wrapper script
# ACTIVATION WORDS: sync to gcs, sync to google cloud, download from gcs
#

# Change to script directory
cd "$(dirname "$0")"

# Run the Python sync script
python3 tools_utilities/gcs_sync_to_google_cloud.py "$@"
