
    def get_status(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current status of all brain modules.
        Returns:
            A dictionary containing the status of each module.
        """
        return {
            "current_stage": self.current_stage,
            "hippocampus_stats": self.hippocampus.get_memory_stats(),
            "thalamus_info_sources": len(self.thalamus.information_sources),
            "imitator_status": {
                "exploration_rate": getattr(self.rl_agent, "exploration_rate", None),
                "num_states_learned": len(self.rl_agent.q_table)
            },
            "dopamine_system_status": {
                "current_level": self.dopamine_system.current_dopamine_level
            },
            "world_model_status": {
                "model_error": np.mean(np.abs(self.world_model.transition_model))
            },
            "meta_controller_status": {
                "intrinsic_weight": self.meta_controller.intrinsic_weight
            },
            "working_memory_status": self.working_memory.get_status(),
            "global_workspace_content": self.global_workspace.get_broadcast(),
            "cerebellum_status": {
                "coordination_noise": self.cerebellum.coordination_noise
            },
            "proto_cortex_mean_activity": self.proto_cortex.mean_activity(),
            "sleep_engine_status": self.sleep_engine.get_sleep_summary(),
            "salience_network_attention": self.salience_network.get_attention_weights().tolist(),
            "dmn_status": self.dmn.get_status(),
            "dna_controller_metrics": self.dna_controller.get_performance_metrics(),
            "llm_ik_stats": self.llm_ik.get_learning_stats() if hasattr(self, 'llm_ik') else {},
            "llm_manipulation_stats": self.llm_manipulation_planner.get_planning_stats() if hasattr(self, 'llm_manipulation_planner') else {},
            "training_data_status": {
                "ik_solutions_loaded": len(self._ik_training_data.get('solutions', [])),
                "manipulation_demos_loaded": len(self._manipulation_training_data.get('demonstrations', [])),
                "unified_dataset_samples": self._unified_training_data.get('dataset_statistics', {}).get('total_training_samples', 0)
            },
            "safety_guardian_status": self.safety_guardian.get_status(),
            "brain_stem_status": self.brain_stem.get_status(),
            "adapters": {
                "ruckig": (self.ruckig.get_status() if self.ruckig else {"available": False}),
                "toppra": (self.toppra.get_status() if self.toppra else {"available": False}),
                "ompl": (self.ompl.get_status() if self.ompl else {"available": False}),
                "towr": (self.towr.get_status() if self.towr else {"available": False}),
                "ocs2": (self.ocs2.get_status() if self.ocs2 else {"available": False}),
                "visp": (self.visp.get_status() if self.visp else {"available": False}),
                "gtsam": (self.gtsam.get_status() if self.gtsam else {"available": False}),
                "rtabmap": (self.rtabmap.get_status() if self.rtabmap else {"available": False}),
                "pcl": (self.pcl.get_status() if self.pcl else {"available": False}),
                "pinocchio": (self.pin.get_status() if self.pin else {"available": False}),
                "dart": (self.dart.get_status() if self.dart else {"available": False}),
                "drake": (self.drake.get_status() if self.drake else {"available": False}),
                "sophus": (self.sophus.get_status() if self.sophus else {"available": False}),
                "manif": (self.manif.get_status() if self.manif else {"available": False}),
                "spatialmath": (self.spatialmath.get_status() if self.spatialmath else {"available": False}),
                "casadi": (self.casadi.get_status() if self.casadi else {"available": False}),
                "osqp": (self.osqp_backend.get_status() if self.osqp_backend else {"available": False}),
                "ceres": (self.ceres.get_status() if self.ceres else {"available": False}),
                "qpsolvers": (self.qps.get_status() if self.qps else {"available": False}),
                "nlopt": (self.nlopt.get_status() if self.nlopt else {"available": False}),
                "ipopt": (self.ipopt.get_status() if self.ipopt else {"available": False}),
                "proxsuite": (self.proxsuite.get_status() if self.proxsuite else {"available": False}),
                "pagmo": (self.pagmo.get_status() if self.pagmo else {"available": False}),
                "pymoo": (self.pymoo.get_status() if self.pymoo else {"available": False}),
                "trajopt": (self.trajopt.get_status() if self.trajopt else {"available": False}),
                "topico": (self.topico.get_status() if self.topico else {"available": False}),
                "fuse": (self.fuse.get_status() if self.fuse else {"available": False}),
            }
        }
