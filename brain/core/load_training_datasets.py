
    def _load_training_datasets_if_needed(self):
        """
        Load training datasets on-demand to avoid slow startup.
        """
        if self._ik_training_data is not None:
            return # Already loaded

        try:
            print("📚 Lazily loading training datasets...")
            
            # Load IK training data
            self._ik_training_data = self.dataset_integration.load_ik_training_data()
            print(f"   ✅ IK Solutions: {len(self._ik_training_data['solutions'])}")
            
            # Load manipulation training data  
            self._manipulation_training_data = self.dataset_integration.load_manipulation_training_data()
            print(f"   ✅ Manipulation Demos: {len(self._manipulation_training_data['demonstrations'])}")
            
            # Create unified dataset
            self._unified_training_data = self.dataset_integration.create_unified_training_dataset()
            print(f"   ✅ Unified Dataset: {self._unified_training_data['dataset_statistics']['total_training_samples']} samples")
            
            # Get training recommendations
            self._training_recommendations = self.dataset_integration.get_training_recommendations()
            print(f"   💡 Training Recommendations: {len(self._training_recommendations['curriculum_order'])} phases")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load training datasets: {e}")
            self._ik_training_data = {"solutions": []}
            self._manipulation_training_data = {"demonstrations": []}
            self._unified_training_data = {"dataset_statistics": {"total_training_samples": 0}}
            self._training_recommendations = {"curriculum_order": []}
