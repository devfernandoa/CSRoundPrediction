import pandas as pd
from sklearn.base import clone

class MiniModels():
    def saveData(self, X, Y, model, n_splits=5, verbose=False):
        self.X = X
        self.Y = Y
        self.model = model
        self.n_splits = n_splits
        self.models = {}
        self.verbose = verbose
    
    def concatData(self, X, Y):
        self.Data = pd.concat([X, Y], axis=1)
    
    def splitData(self):
        part_size = len(self.Data) // self.n_splits 
        self.Data = self.Data.sort_values(by='time_left', ascending=False)
        self.parts = [self.Data[i*part_size:(i+1)*part_size] for i in range(self.n_splits)]

        if len(self.Data) % self.n_splits != 0:
            self.parts[-1] = pd.concat([self.parts[-1], self.Data[self.n_splits*part_size:]], axis=0)
        
        # Display the self.parts:
        print("Self.parts:")
        for i, part in enumerate(self.parts):
            print(f"Part {i} size: {(part.shape)}, time_left: {part['time_left'].max()} - {part['time_left'].min()}")

    def fit(self, X, Y, model, n_splits=5, verbose=False):
        self.saveData(X, Y, model, n_splits, verbose)
        self.concatData(self.X, self.Y)
        self.splitData()
        if self.n_splits == 1:
            self.model.fit(self.X, self.Y)
        else:
            for i in range(1, self.n_splits-1):
                print(f"Training model, intervals {i-1}, {i}, {i+1}") if self.verbose else None
                interval_0 = pd.concat([self.parts[i-1], self.parts[i]], axis=0)
                interval_1 = self.parts[i]
                interval_2 = pd.concat([self.parts[i], self.parts[i+1]], axis=0)
                # Time_left range
                max_time = interval_0['time_left'].max()
                min_time = round(interval_2['time_left'].min() - 0.01,2)
                
                print(f"Time range: {max_time} - {min_time}") if self.verbose else None
                
                # interval 0
                X_0 = interval_0.drop('round_winner', axis=1)
                Y_0 = interval_0['round_winner']
                model_0 = clone(self.model)
                model_0.fit(X_0, Y_0)
                
                # interval 1
                X_1 = interval_1.drop('round_winner', axis=1)
                Y_1 = interval_1['round_winner']
                model_1 = clone(self.model)  
                model_1.fit(X_1, Y_1)
                
                # interval 2
                X_2 = interval_2.drop('round_winner', axis=1)
                Y_2 = interval_2['round_winner']
                model_2 = clone(self.model)
                model_2.fit(X_2, Y_2)
                
                self.models[i-1] = {
                                    "time_range": (max_time, min_time),
                                    "models": [model_0, model_1, model_2]
                                }
                
            print("Models trained")
                
    def select_trio(self,time_left):
        best_model = None
        best_time = None
        best_diff = float('inf') # Initialize with infinity
        print(f"Time_left: {time_left}") if self.verbose else None
        for idx, model_info in self.models.items():
            max_time, min_time = model_info["time_range"]
            print(f"Model {idx} time range: {max_time} - {min_time}") if self.verbose else None
            if (time_left <= max_time) and (time_left >= min_time):
                avg_time = (max_time + min_time) / 2
                diff = abs(time_left - avg_time)
                if diff < best_diff:
                    best_diff = diff
                    best_model = model_info["models"]
                    best_time = (max_time, min_time)
        
        print(f"Selected trio for time_left: {time_left} was {best_time}") if self.verbose else None
        print(f"Best model: {best_model}") if self.verbose else None
        return best_model
    
    def predict(self, X):
        predictions = []
        if self.n_splits == 1:
            return self.model.predict(X)
        for _, row in X.iterrows():
            time_left = row['time_left']
            selected_models = self.select_trio(time_left)
            if selected_models == []:
                raise ValueError(f"Could not find suitable models for time_left: {time_left}")
                
            drop_time_left = False 
            if drop_time_left:
                row = row.drop('time_left').values.reshape(1, -1)
            
            votes = [model.predict(row.to_frame().T) for model in selected_models] # Using to_frame method to convert the to a single line DataFrame. The method to_frame transform the row indexes to Columns, then it's necessary to use .T method to calculate the transpose of the DataFrame
            predictions.append(1 if votes.count(1) > votes.count(0) else 0)
        
        return predictions
            
        
        