import numpy as np

class CentralPatternGenerator:
    """
    A two-level CPG for generating a more realistic walking gait with distinct
    stance and swing phases.
    """
    def __init__(self, num_legs=2, num_joints_per_leg=4):
        self.num_legs = num_legs
        self.num_joints_per_leg = num_joints_per_leg # hip_x, hip_z, hip_y, knee
        self.time = 0.0
        
        # Rhythm Generation Layer
        self.frequency = 0.6
        self.phase_offset = np.pi # Legs are perfectly out of phase

        # Pattern Formation Layer Parameters
        self.hip_amplitude = 0.8
        self.knee_amplitude = 0.8
        self.stance_hip_angle = 0.2
        self.swing_hip_angle = -0.4
        self.stance_knee_angle = 0.1
        self.swing_knee_angle = 0.8 # More pronounced knee bend during swing

    def step(self, dt=0.01):
        """Advance the CPG by one time step."""
        self.time += dt
        joint_angles = np.zeros((self.num_legs, self.num_joints_per_leg))

        for leg in range(self.num_legs):
            # Rhythm generator for this leg
            phase = 2 * np.pi * self.frequency * self.time + (leg * self.phase_offset)
            rhythm_signal = np.sin(phase)

            # Pattern Formation based on rhythm
            if rhythm_signal > 0: # Swing Phase
                hip_y_angle = self.swing_hip_angle * rhythm_signal
                knee_angle = self.swing_knee_angle * rhythm_signal
            else: # Stance Phase
                hip_y_angle = self.stance_hip_angle * -rhythm_signal
                knee_angle = self.stance_knee_angle * -rhythm_signal
            
            # For now, keep other hip joints simple
            hip_x_angle = 0.0
            hip_z_angle = 0.0

            joint_angles[leg, 0] = hip_x_angle
            joint_angles[leg, 1] = hip_z_angle
            joint_angles[leg, 2] = hip_y_angle
            joint_angles[leg, 3] = knee_angle
                
        # The CPG should output patterns for 8 leg joints
        # (2 legs * 4 joints/leg)
        return joint_angles.flatten()
