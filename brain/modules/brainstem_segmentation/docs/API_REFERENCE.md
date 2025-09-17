# Brainstem Segmentation API Reference

*Version 1.0 - 2025-09-17*

## Base URL
```
http://localhost:8080/api
```

## Authentication
Currently no authentication required. In production, implement API key authentication.

## Endpoints

### POST /segment
Upload and segment brain volume.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Brain volume file (.nii.gz format)
- Max size: 100MB

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "statistics": {
    "total_voxels": 2097152,
    "segmented_voxels": 67108,
    "segmentation_coverage": 0.032,
    "num_regions": 12,
    "region_counts": {
      "class_0": 2030044,
      "class_1": 15432,
      "class_2": 8765,
      "class_3": 12890
    }
  },
  "download_url": "/api/download/550e8400-e29b-41d4-a716-446655440000",
  "visualization_url": "/api/visualize/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses:**
- `413`: File too large
- `400`: Invalid file format
- `500`: Segmentation failed

### GET /visualize/{request_id}
Generate 3-D visualization of segmentation.

**Parameters:**
- `view_type` (query): "nuclei" or "subdivisions" (default: "nuclei")
- `opacity` (query): 0.0-1.0 (default: 0.7)

**Response:** Plotly figure JSON for web rendering

**Example:**
```bash
curl "http://localhost:8080/api/visualize/550e8400-e29b-41d4-a716-446655440000?view_type=subdivisions&opacity=0.8"
```

### GET /slice/{request_id}
Get 2-D slice view of segmentation.

**Parameters:**
- `axis` (query): 0, 1, or 2 (default: 2)
- `slice_index` (query): Slice number (default: middle slice)

**Response:** Plotly figure JSON for 2-D slice visualization

### GET /download/{request_id}
Download segmentation results.

**Response:** 
- Content-Type: `application/gzip`
- File: `brainstem_segmentation_{request_id}.nii.gz`

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "brainstem_segmentation_api", 
  "version": "1.0.0"
}
```

### GET /metrics
Get current Prometheus metrics.

**Response:**
```json
{
  "total_runs": 150,
  "successful_runs": 147,
  "latest_overall_dice": 0.89,
  "latest_dice_drift": 0.02
}
```

## Python SDK

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
import requests
import numpy as np
import nibabel as nib

# Load brain volume
img = nib.load("brain_volume.nii.gz")

# Upload for segmentation
with open("brain_volume.nii.gz", "rb") as f:
    response = requests.post(
        "http://localhost:8080/api/segment",
        files={"file": f}
    )

result = response.json()
request_id = result["request_id"]

# Download result
download_response = requests.get(f"http://localhost:8080/api/download/{request_id}")
with open("segmentation_result.nii.gz", "wb") as f:
    f.write(download_response.content)
```

### Advanced Usage
```python
from brain.modules.brainstem_segmentation import (
    BrainstemInferenceEngine, 
    InferenceConfig,
    BrainstemVisualizer
)

# Direct inference
config = InferenceConfig(device="cuda", batch_size=4)
engine = BrainstemInferenceEngine(config)
segmentation = engine.segment_volume(volume)

# Visualization
visualizer = BrainstemVisualizer()
fig_data = visualizer.create_3d_visualization(segmentation, view_type="nuclei")
```

## Rate Limits

### Current Limits
- **Concurrent requests**: 10
- **File uploads**: 100MB max
- **Request rate**: 60/minute per IP

### Scaling
For higher throughput:
1. Deploy multiple API instances behind load balancer
2. Use Redis for distributed caching
3. Enable batch inference processing

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad request (invalid file format)
- `413`: Payload too large
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Service unavailable

### Error Response Format
```json
{
  "error": "error_code",
  "message": "Human readable error message",
  "details": "Additional technical details"
}
```

## Monitoring Integration

### Prometheus Metrics
- `brainstem_segmentation_latency_seconds`
- `brainstem_segmentation_runs_total`
- `brainstem_segmentation_success_total`
- `brainstem_segmentation_overall_dice`
- `brainstem_segmentation_dice_drift`
- `brainstem_segmentation_boundary_error_um`

### Grafana Dashboard
Import `management/configurations/project/grafana_dashboards/brainstem_segmentation.json`

### Log Format
```
2025-09-17 12:34:56 INFO [request_id] Segmentation completed: dice=0.89, latency=1.2s
```

---

*For additional support, consult the User Manual or contact the development team.*
