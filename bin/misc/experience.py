class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __str__(self):
        return 'State : {0}\nAction : {1}\nReward : {2}\nNext State : {3}\nDone : {4}\n' \
               '---------------------------------------------------------------------'.format(self.state,
                                                                                              self.action,
                                                                                              self.reward,
                                                                                              self.next_state,
                                                                                              self.done)
