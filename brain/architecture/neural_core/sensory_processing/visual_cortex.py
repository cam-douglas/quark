

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
try:
    import cv2
except Exception:  # optional dependency guard for headless/import audit
    cv2 = None  # type: ignore
from typing import Dict, Any, Optional
import mujoco

class VisualCortex:
    """
    Processes visual data by rendering the scene and identifying salient objects.
    """
    def __init__(self, embodiment: Any):
        self.embodiment = embodiment
        if self.embodiment is None:
            raise ValueError("VisualCortex requires an embodiment.")

        # Get all required components from the embodiment
        self.viewer = self.embodiment.viewer
        self.model = self.embodiment.model
        self.data = self.embodiment.data

        if not all([self.viewer, self.model, self.data]):
             raise ValueError("Embodiment is missing viewer, model, or data.")

        # Create a scene and context for offscreen rendering
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        print("👁️ Visual Cortex initialized (MuJoCo Offscreen Renderer).")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Renders the current simulation state to an offscreen buffer and returns it as a NumPy array.
        """
        # Ensure the viewer window is available and running before rendering
        if not self.viewer or not self.viewer.is_running():
            return None

        try:
            viewport = mujoco.MjrRect(0, 0, self.viewer.width, self.viewer.height)

            # Update scene and render
            mujoco.mjv_updateScene(self.model, self.data, self.viewer.user_scn, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)

            # Read pixels
            rgb_buffer = np.zeros((self.viewer.height, self.viewer.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb_buffer, None, viewport, self.context)

            # MuJoCo renders with the origin at the bottom-left, so we need to flip it vertically
            return np.flipud(rgb_buffer)
        except Exception:
            # print(f"Error in get_frame: {e}")
            return None

    def process_image(self, image: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """
        Processes an image to find the largest salient object.
        """
        if image is None or image.size == 0:
            return None

        image_height, image_width, _ = image.shape
        # The frame is RGB, convert to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
             return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w / 2
        center_y = y + h / 2

        normalized_x = (center_x - image_width / 2) / (image_width / 2)
        normalized_y = (center_y - image_height / 2) / (image_height / 2)
        detection_area = (w * h) / (image_height * image_width)

        return {
            "x": normalized_x,
            "y": normalized_y,
            "area": detection_area
        }

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Renders an image from the simulation, processes it, and returns the findings.
        """
        frame = self.get_frame()

        if frame is not None:
            object_location = self.process_image(frame)
            return {
                "object_location": object_location,
                "processed_features": frame
            }

        return {"object_location": None, "processed_features": np.array([])}
