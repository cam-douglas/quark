
    def set_viewer(self, viewer: Any):
        """
        Sets the viewer for the VisualCortex after it has been initialized.
        """
        if viewer and self.embodiment:
            # Pass the entire embodiment, which has model, data, and viewer
            if self.visual_cortex:
                self.visual_cortex.set_embodiment(self.embodiment)
            else:
                 # If bio-sim did not produce a visual cortex, we can create one here
                 # or leave it disabled. For now, we'll create it if a viewer is provided.
                self.visual_cortex = VisualCortex(embodiment=self.embodiment)
                print("   - Visual Cortex dynamically instantiated after viewer was set.")
        else:
            print("⚠️ Warning: No viewer/embodiment provided. VisualCortex is disabled.")
