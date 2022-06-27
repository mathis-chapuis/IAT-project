class AgentInterface:
    def select_action(self, state):
        pass

    def select_greedy_action(self, state):
        return self.select_action(state)
