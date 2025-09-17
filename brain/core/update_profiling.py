
    def _update_profiling_data(self, t_start, t_vision, t_sensory, t_proto, t_ppo_select, t_motor, t_cognitive, t_learning):
        self.profiling_data['vision'] = self.profiling_data.get('vision', 0) + (t_vision - t_start)
        self.profiling_data['sensory'] = self.profiling_data.get('sensory', 0) + (t_sensory - t_vision)
        self.profiling_data['proto'] = self.profiling_data.get('proto', 0) + (t_proto - t_sensory)
        self.profiling_data['ppo_select'] = self.profiling_data.get('ppo_select', 0) + (t_ppo_select - t_proto)
        self.profiling_data['motor'] = self.profiling_data.get('motor', 0) + (t_motor - t_ppo_select)
        self.profiling_data['cognitive'] = self.profiling_data.get('cognitive', 0) + (t_cognitive - t_motor)
        self.profiling_data['learning'] = self.profiling_data.get('learning', 0) + (t_learning - t_cognitive)
        self.profiling_data['total'] = self.profiling_data.get('total', 0) + (t_learning - t_start)
        self.profiling_counter += 1

        if self.profiling_counter >= 500:
            avg_total = (self.profiling_data['total'] / self.profiling_counter) * 1000
            print(f"\n--- Brain Step Profile (avg over {self.profiling_counter} steps): {avg_total:.2f} ms ---")
            for key, value in self.profiling_data.items():
                if key != 'total':
                    avg_time = (value / self.profiling_counter) * 1000
                    percentage = (avg_time / avg_total) * 100 if avg_total > 0 else 0
                    print(f"  - {key:<12}: {avg_time:.3f} ms ({percentage:.1f}%)")
            self.profiling_counter = 0
            self.profiling_data = {}
