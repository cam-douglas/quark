#!/bin/bash
# Manual trigger for workspace maintenance
echo "🔧 Running manual workspace maintenance..."
cd "/Users/camdouglas/quark"
bash "/Users/camdouglas/quark/tools_utilities/scripts/cron_maintenance_wrapper.sh"
echo "✅ Manual maintenance completed. Check logs at: "
