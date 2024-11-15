import pandas as np
class MiniModels():
    def saveData(self, X, Y, model, n_splits=5):
        self.X = X
        self.Y = Y
        self.model = model
        self.n_splits = n_splits
    
    def concatData(self, X, Y):
        self.Data = np.concat([X, Y], axis=1)
    
    def splitData(self):
        part_size = len(self.Data) // self.n_splits 
        self.Data = self.Data.sort_values(by='time_left', ascending=False)
        parts = [self.Data[i*part_size:(i+1)*part_size] for i in range(self.n_splits)]

        if len(self.Data) % self.n_splits != 0:
            parts[-1] = np.concat([parts[-1], self.Data[self.n_splits*part_size:]], axis=0)
        
        # Display the parts:
        for i, part in enumerate(parts):
            print(f"Part {i} size: {len(part)}, time_left: {part['time_left'].max()} - {part['time_left'].min()}")

    def fit(self, X, Y, model, n_splits=5):
        self.saveData(X, Y, model, n_splits)
        self.concatData(self.X, self.Y)
        self.splitData()