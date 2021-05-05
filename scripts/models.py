#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class ClassifierModel:
    # global imports
    import time
    t = time
        
    def __init__(self, X, y, model, class_weight=None, random_state=2):
        """Initialize each class attributes"""
        self.X = X
        self.y = y
        self.model = model
        self.class_weight = class_weight
        self.random_state = random_state
        
    def __str__(self):
        return (f"ClassifierModel( X={self.X}, y={self.y}, model={self.model}, random_state={self.random_state})")
        
    def __repr__(self):
        """Return string representation of object."""
        return str(self)
        
    def _initialize_model(self):
        return self.model(class_weight = self.class_weight,random_state = self.random_state)
    
    def _train_model(self):
        from sklearn.model_selection import train_test_split
        self.train_test_split = train_test_split
        X_train, X_test, y_train, y_test = self.train_test_split(self.X, self.y, random_state=self.random_state)
        # initialize model
        model = self._initialize_model()
        return X_train, X_test, y_train, y_test, model.fit(X_train, y_train)
        
    def print_accuracy_score(self, time=t):
        """Returns string object representing """
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        import warnings
        # silence warnings
        warnings.filterwarnings('ignore')
        self.accuracy_score = accuracy_score
        self.f1_score = f1_score
        self.classification_report = classification_report
        self.t = time
        start = self.t.time()
        X_train, X_test, y_train, y_test, model = self._train_model()
        model
        # make predictions
        y_pred = model.predict(X_test)
        score = self.accuracy_score(y_pred, y_test)
        f1_score = self.f1_score(y_test, y_pred)
        duration = self.t.time() - start
        if duration > 60:
            print(f"Completed in {round(duration/60)} minutes", flush=True)
        else:
            print(f"Completed in {round(duration, 2)} seconds", flush=True)
        print(f"Accuracy of {self.model.__name__} model: {score: 0.2%}", flush=True)
        print(f"F1 score for the {self.model.__name__} model: {f1_score: 0.3}\n", flush=True)
        print(self.classification_report(y_test, y_pred))
        
    def print_cross_val_score(self, time=t):
        from collections import namedtuple
        import warnings
        # silence warnings
        warnings.filterwarnings('ignore')
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
        print(f"Cross_Val {self.model.scoring} value: {self.model.best_score_: 0.2%}\nBest Paramters:\n{self.model.best_params_}")
        return Metric(self.model.best_score_, duration)
    
    def plot_pr_curve(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import average_precision_score, plot_precision_recall_curve
        import matplotlib.pyplot as plt
        import warnings
        # silence warnings
        warnings.filterwarnings('ignore')
        self.train_test_split = train_test_split
        self.average_precision_score = average_precision_score
        self.plot_precision_recall_curve = plot_precision_recall_curve
        # train model
        X_train, X_test, y_train, y_test, model = self._train_model()
        model
    
        y_score = model.decision_function(X_test)
        average_precision = self.average_precision_score(y_test, y_score)
        
        plt.rcParams["figure.figsize"] = (12, 8)
        axis_format_prcnt = plt.FuncFormatter(lambda x, loc: f"{x:,.1%}")
        disp = self.plot_precision_recall_curve(model, X_test, y_test)
        disp.ax_.get_xaxis().set_major_formatter(axis_format_prcnt)
        disp.ax_.get_yaxis().set_major_formatter(axis_format_prcnt)
        disp.ax_.set_xlabel('Recall (Positive label: 1)')
        disp.ax_.set_ylabel('Precision (Positive label: 1)')
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           f'AP={average_precision:,.1%}');