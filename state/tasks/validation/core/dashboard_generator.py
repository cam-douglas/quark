#!/usr/bin/env python3
"""
Dashboard Generator Module
==========================
Generates validation dashboards and visualizations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class DashboardGenerator:
    """Generate validation dashboards and reports."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.dashboards_dir = validation_root / "dashboards"
        self.evidence_dir = validation_root / "evidence"
        self.dashboards_dir.mkdir(exist_ok=True)
    
    def generate_html_dashboard(self) -> Path:
        """Generate HTML dashboard for validation results."""
        dashboard_path = self.dashboards_dir / "validation_dashboard.html"
        
        # Collect metrics from all runs
        all_metrics = self._collect_all_metrics()
        
        html_content = self._generate_html_template(all_metrics)
        
        with open(dashboard_path, "w") as f:
            f.write(html_content)
        
        print(f"✅ HTML dashboard generated: {dashboard_path}")
        return dashboard_path
    
    def _collect_all_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics from all evidence runs."""
        metrics_list = []
        
        for run_dir in sorted(self.evidence_dir.iterdir()):
            if run_dir.is_dir():
                metrics_file = run_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                        metrics["run_id"] = run_dir.name
                        metrics_list.append(metrics)
        
        return metrics_list
    
    def _generate_html_template(self, metrics: List[Dict]) -> str:
        """Generate HTML dashboard template."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Quark Validation Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }
        .run-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .run-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .run-table td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        .status-pass { color: #28a745; font-weight: bold; }
        .status-fail { color: #dc3545; font-weight: bold; }
        .status-pending { color: #ffc107; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Quark Validation Dashboard</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Total Validation Runs</div>
                <div class="metric-value">""" + str(len(metrics)) + """</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Latest Run</div>
                <div class="metric-value">""" + (metrics[-1]["run_id"] if metrics else "N/A") + """</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Average ECE</div>
                <div class="metric-value">""" + self._calculate_avg_ece(metrics) + """</div>
            </div>
        </div>
        
        <h2>Recent Validation Runs</h2>
        <table class="run-table">
            <thead>
                <tr>
                    <th>Run ID</th>
                    <th>Timestamp</th>
                    <th>Checklist</th>
                    <th>KPIs Passed</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add rows for each run
        for metric in reversed(metrics[-10:]):  # Last 10 runs
            run_id = metric.get("run_id", "unknown")
            timestamp = metric.get("timestamp", "N/A")
            checklist = metric.get("checklist", "N/A")
            
            # Count passed KPIs
            passed = 0
            total = 0
            if "kpis" in metric:
                for kpi_name, kpi_data in metric["kpis"].items():
                    if isinstance(kpi_data, dict):
                        total += 1
                        if kpi_data.get("status") == "success":
                            passed += 1
            
            status_class = "status-pass" if passed == total and total > 0 else "status-fail"
            status = "PASS" if passed == total and total > 0 else "FAIL"
            
            html += f"""
                <tr>
                    <td>{run_id}</td>
                    <td>{timestamp[:19] if len(timestamp) > 19 else timestamp}</td>
                    <td>{checklist}</td>
                    <td>{passed}/{total}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
            Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _calculate_avg_ece(self, metrics: List[Dict]) -> str:
        """Calculate average ECE across runs."""
        ece_values = []
        
        for metric in metrics:
            if "calibration" in metric and "ECE" in metric["calibration"]:
                try:
                    ece_values.append(float(metric["calibration"]["ECE"]))
                except (ValueError, TypeError):
                    pass
        
        if ece_values:
            avg_ece = sum(ece_values) / len(ece_values)
            return f"{avg_ece:.3f}"
        return "N/A"
    
    def generate_grafana_config(self) -> Path:
        """Generate Grafana dashboard configuration."""
        config_path = self.dashboards_dir / "grafana_dashboard.json"
        
        # Load existing panels if available
        panels_file = self.dashboards_dir / "grafana_panels.json"
        if panels_file.exists():
            with open(panels_file) as f:
                panels = json.load(f)
        else:
            panels = self._create_default_panels()
        
        dashboard_config = {
            "dashboard": {
                "title": "Quark Validation Metrics",
                "panels": panels,
                "refresh": "30s",
                "time": {
                    "from": "now-7d",
                    "to": "now"
                },
                "version": 1
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(dashboard_config, f, indent=2)
        
        print(f"✅ Grafana config generated: {config_path}")
        return config_path
    
    def _create_default_panels(self) -> List[Dict]:
        """Create default Grafana panels."""
        return [
            {
                "id": 1,
                "title": "Calibration (ECE)",
                "type": "graph",
                "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
            },
            {
                "id": 2,
                "title": "KPI Pass Rate",
                "type": "graph",
                "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
            },
            {
                "id": 3,
                "title": "Biological Fidelity Metrics",
                "type": "table",
                "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8}
            }
        ]
    
    def generate_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends across validation runs."""
        metrics = self._collect_all_metrics()
        
        if len(metrics) < 2:
            return {"status": "insufficient_data"}
        
        trends = {
            "total_runs": len(metrics),
            "kpi_trends": {},
            "calibration_trend": [],
            "pass_rate_trend": []
        }
        
        # Track KPI values over time
        kpi_history = {}
        
        for metric in metrics:
            if "kpis" in metric:
                for kpi_name, kpi_data in metric["kpis"].items():
                    if isinstance(kpi_data, dict) and "value" in kpi_data:
                        if kpi_name not in kpi_history:
                            kpi_history[kpi_name] = []
                        kpi_history[kpi_name].append({
                            "run_id": metric.get("run_id"),
                            "value": kpi_data["value"]
                        })
        
        # Calculate trends
        for kpi_name, history in kpi_history.items():
            if len(history) >= 2:
                # Simple linear trend
                values = [float(h["value"]) for h in history if isinstance(h["value"], (int, float))]
                if len(values) >= 2:
                    trend = "improving" if values[-1] > values[0] else "declining"
                    trends["kpi_trends"][kpi_name] = {
                        "direction": trend,
                        "latest": values[-1],
                        "change": values[-1] - values[0]
                    }
        
        return trends
