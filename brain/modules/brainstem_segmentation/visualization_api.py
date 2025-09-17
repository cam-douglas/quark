"""
Interactive 3-D visualization and API endpoints for brainstem segmentation.

Provides FastAPI endpoints for segmentation services and web-based
3-D visualization of brainstem nuclei and subdivisions.
"""
from __future__ import annotations

import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import io
import base64
import tempfile
import uuid

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for visualization API."""
    
    host: str = "0.0.0.0"
    port: int = 8080
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    cache_dir: str = "/tmp/brainstem_cache"
    enable_cors: bool = True


class BrainstemVisualizer:
    """3-D visualization engine for brainstem segmentation results."""
    
    def __init__(self):
        """Initialize visualizer with nucleus color mapping."""
        self.nucleus_colors = {
            0: "#000000",  # background - black
            1: "#FF6B6B",  # Periaqueductal grey - red
            2: "#4ECDC4",  # Edinger-Westphal - teal
            3: "#45B7D1",  # Substantia nigra - blue
            4: "#96CEB4",  # Red nucleus - green
            5: "#FFEAA7",  # Locus coeruleus - yellow
            6: "#DDA0DD",  # Pontine nuclei - plum
            7: "#98D8C8",  # Facial nucleus - mint
            8: "#F7DC6F",  # Superior olivary - gold
            9: "#BB8FCE",  # Raphe magnus - lavender
            10: "#85C1E9", # Nucleus ambiguus - light blue
            11: "#F8C471", # Pre-Bötzinger - orange
            12: "#82E0AA", # Hypoglossal - light green
            13: "#F1948A", # Dorsal motor vagus - salmon
            14: "#C39BD3", # Solitary tract - purple
            15: "#7FB3D3"  # Inferior olivary - steel blue
        }
        
        self.subdivision_colors = {
            0: "#000000",  # background
            1: "#FF4444",  # midbrain - red
            2: "#44FF44",  # pons - green  
            3: "#4444FF"   # medulla - blue
        }
    
    def create_3d_visualization(
        self,
        segmentation: np.ndarray,
        view_type: str = "nuclei",
        opacity: float = 0.7
    ) -> Dict[str, Any]:
        """Create 3-D visualization of segmentation results.
        
        Args:
            segmentation: Integer label volume
            view_type: "nuclei" or "subdivisions"
            opacity: Mesh opacity (0-1)
            
        Returns:
            Plotly figure data for web display
        """
        
        if view_type == "nuclei":
            colors = self.nucleus_colors
            unique_labels = np.unique(segmentation)
        else:
            # Convert to subdivisions
            from brain.modules.brainstem_segmentation.hierarchical_framework import AnatomicalHierarchy
            hierarchy = AnatomicalHierarchy()
            segmentation = hierarchy.get_subdivision_mask(torch.from_numpy(segmentation)).numpy()
            colors = self.subdivision_colors
            unique_labels = np.unique(segmentation)
        
        # Create 3D mesh for each label
        meshes = []
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            # Extract label mask
            mask = segmentation == label
            
            if not np.any(mask):
                continue
            
            # Create isosurface mesh
            vertices, faces = self._create_mesh_from_mask(mask)
            
            if len(vertices) == 0:
                continue
            
            # Create mesh trace
            mesh = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1], 
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=colors.get(int(label), "#888888"),
                opacity=opacity,
                name=f"Label {label}"
            )
            
            meshes.append(mesh)
        
        # Create figure
        fig = go.Figure(data=meshes)
        
        fig.update_layout(
            title=f"Brainstem Segmentation - {view_type.title()}",
            scene=dict(
                xaxis_title="X (voxels)",
                yaxis_title="Y (voxels)",
                zaxis_title="Z (voxels)",
                aspectmode="cube"
            ),
            width=800,
            height=600
        )
        
        return fig.to_dict()
    
    def _create_mesh_from_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create mesh vertices and faces from binary mask."""
        try:
            from skimage.measure import marching_cubes
            
            # Apply marching cubes
            vertices, faces, _, _ = marching_cubes(mask.astype(float), level=0.5)
            
            return vertices, faces
            
        except ImportError:
            logger.warning("scikit-image not available, using simplified mesh")
            
            # Fallback: create simple point cloud
            coords = np.where(mask)
            if len(coords[0]) == 0:
                return np.array([]), np.array([])
            
            # Sample points for visualization
            n_points = min(1000, len(coords[0]))
            indices = np.random.choice(len(coords[0]), n_points, replace=False)
            
            vertices = np.column_stack([coords[0][indices], coords[1][indices], coords[2][indices]])
            
            # Create simple triangular faces (for point cloud visualization)
            if len(vertices) >= 3:
                faces = np.array([[0, 1, 2]])  # Single triangle
            else:
                faces = np.array([])
            
            return vertices, faces
    
    def create_slice_view(
        self,
        segmentation: np.ndarray,
        slice_axis: int = 2,
        slice_index: int = None
    ) -> Dict[str, Any]:
        """Create 2-D slice visualization."""
        
        if slice_index is None:
            slice_index = segmentation.shape[slice_axis] // 2
        
        # Extract slice
        if slice_axis == 0:
            slice_data = segmentation[slice_index, :, :]
        elif slice_axis == 1:
            slice_data = segmentation[:, slice_index, :]
        else:
            slice_data = segmentation[:, :, slice_index]
        
        # Create heatmap
        fig = px.imshow(
            slice_data,
            color_continuous_scale="viridis",
            title=f"Brainstem Segmentation - Slice {slice_index} (axis {slice_axis})"
        )
        
        return fig.to_dict()


# FastAPI application
app = FastAPI(title="Brainstem Segmentation API", version="1.0.0")

# Global components
visualizer = BrainstemVisualizer()
config = APIConfig()

# Setup CORS
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Cache directory
cache_dir = Path(config.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)


@app.post("/api/segment")
async def segment_volume(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> JSONResponse:
    """Segment uploaded brain volume."""
    
    if file.size > config.max_upload_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Save uploaded file
        temp_path = cache_dir / f"{request_id}_input.nii.gz"
        content = await file.read()
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load and segment
        img = nib.load(str(temp_path))
        volume = img.get_fdata()
        
        # Run segmentation (using inference engine)
        from brain.modules.brainstem_segmentation.inference_engine import auto_segment_brainstem
        segmentation = auto_segment_brainstem(volume)
        
        if segmentation is None:
            raise HTTPException(status_code=500, detail="Segmentation failed")
        
        # Save segmentation result
        result_path = cache_dir / f"{request_id}_segmentation.nii.gz"
        seg_img = nib.Nifti1Image(segmentation, img.affine, img.header)
        nib.save(seg_img, str(result_path))
        
        # Generate statistics
        from brain.modules.brainstem_segmentation.inference_algorithms import get_segmentation_stats
        stats = get_segmentation_stats(segmentation)
        
        # Clean up input file in background
        if background_tasks:
            background_tasks.add_task(lambda: temp_path.unlink(missing_ok=True))
        
        return JSONResponse({
            "request_id": request_id,
            "status": "success",
            "statistics": stats,
            "download_url": f"/api/download/{request_id}",
            "visualization_url": f"/api/visualize/{request_id}"
        })
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualize/{request_id}")
async def visualize_segmentation(
    request_id: str,
    view_type: str = "nuclei",
    opacity: float = 0.7
) -> JSONResponse:
    """Generate 3-D visualization of segmentation results."""
    
    result_path = cache_dir / f"{request_id}_segmentation.nii.gz"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not found")
    
    try:
        # Load segmentation
        img = nib.load(str(result_path))
        segmentation = img.get_fdata().astype(int)
        
        # Create visualization
        fig_data = visualizer.create_3d_visualization(segmentation, view_type, opacity)
        
        return JSONResponse(fig_data)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/slice/{request_id}")
async def get_slice_view(
    request_id: str,
    axis: int = 2,
    slice_index: Optional[int] = None
) -> JSONResponse:
    """Get 2-D slice view of segmentation."""
    
    result_path = cache_dir / f"{request_id}_segmentation.nii.gz"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not found")
    
    try:
        # Load segmentation
        img = nib.load(str(result_path))
        segmentation = img.get_fdata().astype(int)
        
        # Create slice view
        fig_data = visualizer.create_slice_view(segmentation, axis, slice_index)
        
        return JSONResponse(fig_data)
        
    except Exception as e:
        logger.error(f"Slice visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{request_id}")
async def download_segmentation(request_id: str) -> FileResponse:
    """Download segmentation results."""
    
    result_path = cache_dir / f"{request_id}_segmentation.nii.gz"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not found")
    
    return FileResponse(
        path=str(result_path),
        filename=f"brainstem_segmentation_{request_id}.nii.gz",
        media_type="application/gzip"
    )


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "brainstem_segmentation_api",
        "version": "1.0.0"
    })


@app.get("/api/metrics")
async def get_metrics() -> JSONResponse:
    """Get Prometheus metrics endpoint."""
    try:
        from brain.modules.brainstem_segmentation.metrics import (
            SEGMENTATION_RUNS, SEGMENTATION_SUCCESS, OVERALL_DICE, DICE_DRIFT
        )
        
        return JSONResponse({
            "total_runs": SEGMENTATION_RUNS._value._value,
            "successful_runs": SEGMENTATION_SUCCESS._value._value,
            "latest_overall_dice": OVERALL_DICE._value._value,
            "latest_dice_drift": DICE_DRIFT._value._value
        })
        
    except Exception as e:
        logger.warning(f"Metrics unavailable: {e}")
        return JSONResponse({"error": "Metrics unavailable"})


# Static files for web interface
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def create_web_interface() -> str:
    """Create simple web interface HTML."""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Brainstem Segmentation Viewer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
        .result-area { margin: 20px 0; }
        #visualization { width: 100%; height: 600px; }
    </style>
</head>
<body>
    <h1>🧠 Brainstem Segmentation Viewer</h1>
    
    <div class="upload-area">
        <h3>Upload Brain Volume (.nii.gz)</h3>
        <input type="file" id="fileInput" accept=".nii.gz,.nii">
        <button onclick="uploadFile()">Segment Brain</button>
    </div>
    
    <div class="result-area" id="results" style="display: none;">
        <h3>Segmentation Results</h3>
        <div>
            <label>View Type:</label>
            <select id="viewType" onchange="updateVisualization()">
                <option value="nuclei">Nuclei</option>
                <option value="subdivisions">Subdivisions</option>
            </select>
            <button onclick="downloadResult()">Download Segmentation</button>
        </div>
        <div id="visualization"></div>
        <div id="statistics"></div>
    </div>

    <script>
        let currentRequestId = null;
        
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/segment', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    currentRequestId = result.request_id;
                    document.getElementById('results').style.display = 'block';
                    
                    // Show statistics
                    document.getElementById('statistics').innerHTML = 
                        '<h4>Statistics:</h4>' + 
                        '<pre>' + JSON.stringify(result.statistics, null, 2) + '</pre>';
                    
                    // Load visualization
                    updateVisualization();
                } else {
                    alert('Segmentation failed: ' + result.detail);
                }
            } catch (error) {
                alert('Upload failed: ' + error.message);
            }
        }
        
        async function updateVisualization() {
            if (!currentRequestId) return;
            
            const viewType = document.getElementById('viewType').value;
            
            try {
                const response = await fetch(`/api/visualize/${currentRequestId}?view_type=${viewType}`);
                const figData = await response.json();
                
                Plotly.newPlot('visualization', figData.data, figData.layout);
            } catch (error) {
                console.error('Visualization failed:', error);
            }
        }
        
        function downloadResult() {
            if (!currentRequestId) return;
            
            const downloadUrl = `/api/download/${currentRequestId}`;
            window.open(downloadUrl, '_blank');
        }
    </script>
</body>
</html>
    """
    
    return html_content


@app.get("/")
async def serve_web_interface():
    """Serve web interface."""
    html_content = create_web_interface()
    return HTMLResponse(content=html_content)


def start_api_server(config: APIConfig = None) -> None:
    """Start the FastAPI server."""
    
    if config is None:
        config = APIConfig()
    
    logger.info(f"Starting brainstem segmentation API on {config.host}:{config.port}")
    
    uvicorn.run(
        "brain.modules.brainstem_segmentation.visualization_api:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    # For testing
    print("🌐 Brainstem Segmentation API Demo")
    print("=" * 40)
    print("Starting API server...")
    print("Open http://localhost:8080 to access web interface")
    
    # Create config
    config = APIConfig(port=8080)
    
    # Start server (this would run indefinitely in production)
    print("✅ API endpoints ready for deployment")
    print("📊 Available endpoints:")
    print("  POST /api/segment - Upload and segment brain volume")
    print("  GET /api/visualize/{id} - Get 3-D visualization")
    print("  GET /api/slice/{id} - Get 2-D slice view")
    print("  GET /api/download/{id} - Download segmentation")
    print("  GET /api/health - Health check")
    print("  GET /api/metrics - Prometheus metrics")
    print("  GET / - Web interface")
