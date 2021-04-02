#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class ClassifierModel:
    # global imports
    import time
    t = time
        
    def __init__(self, X, y, model, random_state=2):
        """Initialize each class attributes"""
        self.X = X
        self.y = y
        self.model = model
        self.random_state = random_state
        
    def __str__(self):
        return (f"ClassifierModel( X={self.X}, y={self.y}, model={self.model}, random_state={self.random_state})")
        
    def __repr__(self):
        """Return string representation of object."""
        return str(self)
        
    def print_accuracy_score(self, time=t):
        """Returns string object representing """
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import accuracy_score
        import warnings
        # silence warnings
        warnings.filterwarnings('ignore')
        self.train_test_split = train_test_split
        self.accuracy_score = accuracy_score
        self.t = time
        start = self.t.time()
        # split data into training and test sets
        X_train, X_test, y_train, y_test = self.train_test_split(self.X, self.y, random_state=self.random_state)
        # initialize model
        model = self.model(random_state = self.random_state)
        # train model
        model.fit(X_train, y_train)
        # make predictions
        y_pred = model.predict(X_test)
        score = self.accuracy_score(y_pred, y_test)
        duration = self.t.time() - start
        if duration > 60:
            print(f"Completed in {round(duration/60)} minutes", flush=True)
        else:
            print(f"Completed in {round(duration, 2)} seconds", flush=True)
        return print(f"Accuracy of {self.model.__name__}: {score: 0.2%}")
    
    def print_cross_val_score(self, time=t):
        from collections import namedtuple
        import warnings
        # silence warnings
        warnings.filterwarnings('ignore')
#         self.print_time = print_time
        self.t = time
        start = self.t.time()
        Metric = namedtuple('Metric',['best_score', 'duration'])
        # run model
        self.model.fit(self.X, self.y)
        duration = self.t.time() - start
        if duration > 60:
            print(f"Completed in {round(duration/60)} minutes", flush=True)
        else:
            print(f"Completed in {round(duration, 2)} seconds", flush=True)
        print(f"Cross_Val Accuracy: {self.model.best_score_: 0.2%}\nBest Paramters:\n{self.model.best_params_}")
        return Metric(self.model.best_score_, duration)