# 🛡️ Resource Monitor Setup Guide

This guide will help you set up the Ultimate Resource Authority to run automatically every hour on your Mac M2 Max system.

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)
```bash
cd /Users/camdouglas/quark
python3 install_resource_monitor.py
```

### Option 2: Manual Installation
```bash
cd /Users/camdouglas/quark

# Install the service to run every hour
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --install --interval 1.0 --enable-cloud-offload

# Check that it's running
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --status
```

### Option 3: Custom Configuration
```bash
cd /Users/camdouglas/quark

# Install with custom settings (e.g., every 30 minutes, no cloud offload)
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --install --interval 0.5 --disable-cloud-offload
```

## ⚙️ Service Management

### Check Service Status
```bash
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --status
```

### Start Service (if not running)
```bash
launchctl load ~/Library/LaunchAgents/com.quark.resource.monitor.plist
```

### Stop Service
```bash
launchctl unload ~/Library/LaunchAgents/com.quark.resource.monitor.plist
```

### Uninstall Service
```bash
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --uninstall
```

### View Service Logs
```bash
# Real-time log monitoring
tail -f logs/periodic_monitoring/resource_monitor_$(date +%Y%m%d).log

# View recent service output
cat logs/periodic_monitoring/launchd_stdout.log
```

## 📊 What the Service Does

### Every Hour, the Service:
1. **Monitors Resources**: Checks memory, CPU, temperature, disk I/O
2. **Applies Optimizations**: Reduces parameters if resources are constrained
3. **Offloads to Cloud**: Sends intensive tasks to free cloud platforms
4. **Generates Reports**: Creates detailed JSON reports with trends and recommendations
5. **Logs Activity**: Records all actions and decisions for review

### Automatic Optimizations Include:
- **Memory Management**: Garbage collection, process cleanup, parameter reduction
- **CPU Optimization**: Process priority adjustment, workload scaling
- **Cloud Offloading**: Automatic job submission to Google Colab, Kaggle, etc.
- **Emergency Controls**: Process termination, system protection

### Reports Generated:
- **Hourly Reports**: Detailed resource usage and optimization actions
- **Trend Analysis**: Memory and CPU usage patterns over time
- **Performance Metrics**: Cloud provider performance and success rates
- **Recommendations**: Specific suggestions for system optimization

## 📁 File Locations

### Service Files
- **Service Definition**: `~/Library/LaunchAgents/com.quark.resource.monitor.plist`
- **Monitor Script**: `/Users/camdouglas/quark/brain_modules/resource_monitor/periodic_resource_monitor.py`
- **Resource Authority**: `/Users/camdouglas/quark/brain_modules/resource_monitor/ultimate_resource_authority.py`

### Logs and Reports
- **Service Logs**: `/Users/camdouglas/quark/logs/periodic_monitoring/`
- **Resource Reports**: `/Users/camdouglas/quark/reports/resource_monitoring/`
- **System Logs**: `/Users/camdouglas/quark/logs/resource_monitoring/`

### Configuration
- **Monitor Config**: `/Users/camdouglas/quark/configs/monitoring/`

## 🔧 Configuration Options

### Monitor Intervals
```bash
# Every 30 minutes
--interval 0.5

# Every hour (default)
--interval 1.0

# Every 2 hours
--interval 2.0

# Every 6 hours
--interval 6.0
```

### Cloud Offloading
```bash
# Enable cloud offloading (default)
--enable-cloud-offload

# Disable cloud offloading
--disable-cloud-offload
```

### Logging Levels
```bash
# Debug (verbose)
--log-level DEBUG

# Info (default)
--log-level INFO

# Warning (minimal)
--log-level WARNING

# Error (errors only)
--log-level ERROR
```

## 🔍 Manual Testing

### Run Single Check
```bash
# Run one resource check with reports
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --enable-reports

# Run one check without reports
python3 brain_modules/resource_monitor/periodic_resource_monitor.py
```

### Test Cloud Offloading
```bash
# Run with cloud offloading enabled
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --enable-cloud-offload --enable-reports
```

### Run in Daemon Mode (Testing)
```bash
# Run as daemon for testing (Ctrl+C to stop)
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --daemon --interval 0.1 --enable-cloud-offload
```

## 📈 Monitoring Results

### View Latest Report
```bash
# Find and view the most recent report
ls -la reports/resource_monitoring/ | tail -1
cat reports/resource_monitoring/periodic_report_*.json | jq .
```

### Key Metrics to Monitor
- **Memory Usage**: Should stay below 80% consistently
- **CPU Usage**: Should stay below 85% consistently
- **Temperature**: Should stay below 75°C consistently
- **Optimizations Applied**: Higher numbers indicate resource pressure
- **Cloud Jobs**: Shows how often intensive tasks are offloaded

### Sample Report Output
```json
{
  "metadata": {
    "report_type": "periodic_monitoring",
    "generated_at": "2025-01-21T15:30:00",
    "service_uptime_hours": 24.5,
    "monitor_interval_hours": 1.0
  },
  "service_statistics": {
    "total_checks": 25,
    "optimizations_applied": 3,
    "cloud_jobs_executed": 2,
    "average_memory_percent": 45.2,
    "average_cpu_percent": 32.8,
    "peak_memory_percent": 78.5,
    "peak_cpu_percent": 89.2
  },
  "latest_check": {
    "resource_metrics": {
      "memory_percent": 42.1,
      "cpu_percent": 28.5,
      "temperature_celsius": 52.3
    },
    "optimizations_applied": 0,
    "alerts": [],
    "recommendations": []
  }
}
```

## 🚨 Troubleshooting

### Service Not Running
```bash
# Check if service exists
ls ~/Library/LaunchAgents/com.quark.resource.monitor.plist

# Check launchd status
launchctl list | grep com.quark.resource.monitor

# View error logs
cat logs/periodic_monitoring/launchd_stderr.log
```

### Permission Issues
```bash
# The service may need accessibility permissions
# Go to: System Preferences → Security & Privacy → Privacy → Accessibility
# Add Terminal or the Python executable if prompted
```

### High Resource Usage
```bash
# Check what's using resources
python3 -c "
import psutil
print('Top 5 Memory Users:')
for proc in sorted(psutil.process_iter(['pid', 'name', 'memory_percent']), 
                   key=lambda x: x.info['memory_percent'], reverse=True)[:5]:
    print(f'  {proc.info[\"name\"]}: {proc.info[\"memory_percent\"]:.1f}%')
"
```

### Cloud Offloading Issues
```bash
# Test basic cloud functionality
python3 -c "
from brain_modules.resource_monitor.cloud_offload_authority import CloudOffloadAuthority
authority = CloudOffloadAuthority()
print('Available providers:', list(authority.available_providers.keys()))
print('System status:', authority.get_system_status())
"
```

### Import Errors
```bash
# Test resource monitoring imports
python3 -c "
import sys
sys.path.append('/Users/camdouglas/quark')
try:
    from brain_modules.resource_monitor import create_resource_monitor
    print('✅ Resource monitor available')
except ImportError as e:
    print(f'❌ Import error: {e}')
    # Test basic psutil
    import psutil
    print(f'✅ Basic monitoring available: Memory {psutil.virtual_memory().percent:.1f}%')
"
```

## 🔄 Advanced Usage

### Custom Resource Limits
Create a custom configuration file at `configs/monitoring/custom_config.py`:
```python
from brain_modules.resource_monitor import IntegratedResourceConfig

custom_config = IntegratedResourceConfig(
    # More conservative memory limits
    max_memory_gb=32.0,
    warning_memory_gb=24.0,
    critical_memory_gb=28.0,
    
    # More aggressive cloud offloading
    cloud_offload_memory_threshold=50.0,
    cloud_offload_cpu_threshold=60.0,
    
    # Faster monitoring for real-time workloads
    monitoring_interval=0.5,
)
```

### Integration with Existing Scripts
```python
# In your existing code
from brain_modules.resource_monitor import create_resource_monitor

# Wrap resource-intensive operations
manager = create_resource_monitor(enable_cloud_offload=True)

with manager.integrated_management_context():
    # Your intensive code here
    result = run_brain_simulation()
    
# Check if optimizations were applied
status = manager.get_comprehensive_status()
print(f"Optimizations applied: {len(status.get('recent_decisions', []))}")
```

### Custom Alerts
Add to your scripts to get notified of resource issues:
```python
import subprocess

def send_notification(title, message):
    """Send macOS notification."""
    subprocess.run([
        'osascript', '-e',
        f'display notification "{message}" with title "{title}"'
    ])

# Check resource status
if memory_percent > 85:
    send_notification("Resource Alert", f"High memory usage: {memory_percent:.1f}%")
```

## 📱 Integration Examples

### With Pytest
```python
# conftest.py
import pytest
from brain_modules.resource_monitor import create_resource_monitor

@pytest.fixture(scope="session")
def resource_manager():
    manager = create_resource_monitor()
    manager.start_integrated_management()
    yield manager
    manager.stop_integrated_management()

# Your test files
def test_intensive_operation(resource_manager):
    with resource_manager.integrated_management_context():
        result = run_intensive_test()
        assert result.success
```

### With Jupyter Notebooks
```python
# First cell
from brain_modules.resource_monitor import create_resource_monitor
manager = create_resource_monitor()
manager.start_integrated_management()

# Your notebook cells...

# Last cell
manager.stop_integrated_management()
report = manager.export_comprehensive_report()
print(f"Resource usage report: {report}")
```

## 🎯 Success Indicators

### ✅ Service is Working Correctly When:
- Status command shows service is running
- Hourly reports are being generated
- Memory usage stays below 80% consistently
- CPU usage stays below 85% consistently
- Logs show regular monitoring activity
- No frequent optimization alerts

### ⚠️ May Need Attention When:
- Frequent optimization alerts
- High memory/CPU usage consistently
- Cloud offloading failing repeatedly
- Service stopping unexpectedly
- Reports showing degrading performance trends

### 🚨 Immediate Action Needed When:
- Service won't start
- Memory usage consistently above 90%
- Temperature above 80°C regularly
- System becoming unresponsive
- Critical resource alerts

## 📞 Getting Help

### Log Analysis
```bash
# Check service health
python3 brain_modules/resource_monitor/periodic_resource_monitor.py --status

# View recent logs
tail -50 logs/periodic_monitoring/resource_monitor_$(date +%Y%m%d).log

# Check system resource usage
python3 -c "import psutil; m=psutil.virtual_memory(); print(f'Memory: {m.percent:.1f}% ({m.used/1024**3:.1f}GB/{m.total/1024**3:.1f}GB)')"
```

### Report Issues
If you encounter issues:
1. Check the troubleshooting section above
2. Review the service logs
3. Run a manual test to isolate the problem
4. Check system resource usage independently

### Performance Optimization
The service learns and adapts over time:
- **Adaptive Thresholds**: Automatically adjusts based on your usage patterns
- **Provider Learning**: Remembers which cloud providers work best for different tasks
- **Predictive Management**: Learns to predict resource constraints before they occur

---

## 🎉 You're All Set!

Once installed, the Ultimate Resource Authority will:
- ✅ Monitor your Mac M2 Max resources every hour
- ✅ Automatically optimize performance when needed  
- ✅ Offload intensive tasks to free cloud platforms
- ✅ Generate detailed reports on system health
- ✅ Protect your system from resource exhaustion
- ✅ Learn and adapt to your usage patterns

Your system is now protected with supreme resource authority! 🛡️

