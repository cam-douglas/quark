#!/usr/bin/env python3
"""
SLEEP MONITOR: Monitor brain simulation sleep state
Purpose: Display sleep cycles, consolidation progress, and neural activity during sleep
Inputs: Sleep consolidation engine state
Outputs: Sleep monitoring dashboard
"""

import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from src.core.sleep_consolidation_engine import SleepConsolidationEngine, SleepPhase

class SleepMonitor:
    def __init__(self):
        self.engine = SleepConsolidationEngine()
        self.output_dir = Path("tests/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_data = []
        
    def simulate_sleep_cycle(self, duration_minutes=480):
        """Simulate a complete sleep cycle"""
        print(f"💤 Simulating sleep cycle for {duration_minutes} minutes...")
        
        # Transition to sleep
        self.engine.current_phase = SleepPhase.NREM_1
        print("🌙 Entering NREM-1 (Light Sleep)...")
        
        phases = [SleepPhase.NREM_1, SleepPhase.NREM_2, SleepPhase.NREM_3, SleepPhase.REM]
        phase_names = ["NREM-1", "NREM-2", "NREM-3", "REM"]
        
        for i, (phase, name) in enumerate(zip(phases, phase_names)):
            self.engine.current_phase = phase
            print(f"🔄 Transitioning to {name}...")
            
            # Simulate phase duration
            phase_duration = self.engine.phase_durations[phase]
            for minute in range(int(phase_duration)):
                # Record sleep data
                self.sleep_data.append({
                    'time': len(self.sleep_data),
                    'phase': phase.value,
                    'consolidation': self.engine.consolidation_strengths[phase],
                    'replay': self.engine.replay_probabilities[phase],
                    'neural_activity': np.random.uniform(0.1, 0.8)
                })
                
                # Simulate consolidation
                consolidation_progress = (minute / phase_duration) * self.engine.consolidation_strengths[phase]
                
                if minute % 5 == 0:  # Update every 5 minutes
                    print(f"  {name}: {minute}/{int(phase_duration)}min - Consolidation: {consolidation_progress:.2f}")
                
                time.sleep(0.1)  # Simulate time passing
        
        print("🌅 Sleep cycle completed!")
        
    def create_sleep_dashboard(self):
        """Create sleep monitoring dashboard"""
        print("📊 Creating sleep monitoring dashboard...")
        
        if not self.sleep_data:
            print("⚠️ No sleep data available. Running simulation first...")
            self.simulate_sleep_cycle(60)  # Short cycle for demo
        
        # Extract data
        times = [d['time'] for d in self.sleep_data]
        phases = [d['phase'] for d in self.sleep_data]
        consolidation = [d['consolidation'] for d in self.sleep_data]
        replay = [d['replay'] for d in self.sleep_data]
        neural_activity = [d['neural_activity'] for d in self.sleep_data]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Sleep Phases', 'Consolidation Progress',
                'Replay Activity', 'Neural Activity',
                'Sleep Cycle Overview', 'Memory Consolidation'
            )
        )
        
        # Sleep phases
        phase_colors = {
            'nrem_1': 'lightblue',
            'nrem_2': 'blue',
            'nrem_3': 'darkblue',
            'rem': 'purple'
        }
        
        for phase in set(phases):
            if phase != 'wake':
                phase_times = [t for t, p in zip(times, phases) if p == phase]
                phase_consolidation = [c for c, p in zip(consolidation, phases) if p == phase]
                
                fig.add_trace(
                    go.Scatter(
                        x=phase_times,
                        y=phase_consolidation,
                        mode='lines+markers',
                        name=f'{phase.upper()}',
                        line=dict(color=phase_colors.get(phase, 'gray'), width=3)
                    ),
                    row=1, col=1
                )
        
        # Consolidation progress
        fig.add_trace(
            go.Scatter(
                x=times,
                y=consolidation,
                mode='lines',
                name='Consolidation',
                line=dict(color='green', width=3)
            ),
            row=1, col=2
        )
        
        # Replay activity
        fig.add_trace(
            go.Scatter(
                x=times,
                y=replay,
                mode='lines',
                name='Replay',
                line=dict(color='orange', width=3)
            ),
            row=2, col=1
        )
        
        # Neural activity
        fig.add_trace(
            go.Scatter(
                x=times,
                y=neural_activity,
                mode='lines',
                name='Neural Activity',
                line=dict(color='red', width=3)
            ),
            row=2, col=2
        )
        
        # Sleep cycle overview (heatmap)
        phase_matrix = np.zeros((len(set(phases)), len(times)))
        for i, phase in enumerate(set(phases)):
            for j, t in enumerate(times):
                if phases[j] == phase:
                    phase_matrix[i, j] = 1
        
        fig.add_trace(
            go.Heatmap(
                z=phase_matrix,
                x=times,
                y=list(set(phases)),
                colorscale='Viridis',
                name='Sleep Phases'
            ),
            row=3, col=1
        )
        
        # Memory consolidation (cumulative)
        cumulative_consolidation = np.cumsum(consolidation)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=cumulative_consolidation,
                mode='lines',
                name='Cumulative Consolidation',
                line=dict(color='darkgreen', width=3)
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🧠 Brain Simulation - Sleep Monitoring Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save as HTML
        html_path = self.output_dir / "sleep_monitoring_dashboard.html"
        fig.write_html(str(html_path))
        
        return html_path
    
    def show_sleep_status(self):
        """Display current sleep status"""
        print("\n" + "="*60)
        print("🧠 BRAIN SIMULATION SLEEP STATUS")
        print("="*60)
        
        current_phase = self.engine.current_phase.value.upper()
        print(f"🌙 Current Phase: {current_phase}")
        print(f"⏰ Total Time: {self.engine.total_time:.1f} minutes")
        print(f"🔄 Cycle Count: {self.engine.cycle_count}")
        
        if self.sleep_data:
            latest = self.sleep_data[-1]
            print(f"📊 Latest Consolidation: {latest['consolidation']:.3f}")
            print(f"🔄 Latest Replay: {latest['replay']:.3f}")
            print(f"⚡ Neural Activity: {latest['neural_activity']:.3f}")
        
        print("\n💤 Sleep Phases:")
        for phase in SleepPhase:
            if phase != SleepPhase.WAKE:
                strength = self.engine.consolidation_strengths[phase]
                replay = self.engine.replay_probabilities[phase]
                print(f"  {phase.value.upper()}: Consolidation={strength:.2f}, Replay={replay:.2f}")
        
        print("="*60)
    
    def run_sleep_monitoring(self):
        """Run complete sleep monitoring"""
        print("🚀 Starting sleep monitoring...")
        
        # Show initial status
        self.show_sleep_status()
        
        # Simulate sleep cycle
        self.simulate_sleep_cycle(120)  # 2 hours for demo
        
        # Create dashboard
        dashboard_path = self.create_sleep_dashboard()
        
        # Show final status
        self.show_sleep_status()
        
        print(f"✅ Sleep monitoring completed!")
        print(f"📊 Dashboard saved: {dashboard_path}")
        
        return dashboard_path

if __name__ == "__main__":
    monitor = SleepMonitor()
    monitor.run_sleep_monitoring()
