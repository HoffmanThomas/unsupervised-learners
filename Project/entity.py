import numpy as np

class Entity:
    def __init__(self, data, numEnrichFields):
        self.data = data
        self.state = self.genState(self.data, numEnrichFields)
        self.policy = []

    def genState(self, data, numEnrichFields = 6):
        state = np.array([])

        # interperit missing data as a 0 and present data as 1
        for i in data:
            if i == "":
                state = np.append(state,0)
            else:
                state = np.append(state,1)

        # add the blank values for enrichment fields
        for j in range(0,numEnrichFields):
            state = np.append(state,0)

        return state

    def updatePolicy(self, action):
        self.policy.append(action)
