from sklearn.base import BaseEstimator, ClassifierMixin 
import numpy as np

class SimplePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, kmax,learning_rate=1):
        self.kmax_ = kmax
        self.class_labels_ = None
        self.learning_rate_ = learning_rate
        self.w_ = None # wektor wag
        self.k_ = 0 # licznik krokow


    def fit(self, X, y):
        self.class_labels_ = np.unique(y)
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.class_labels_[0]] = -1 # zakladamy dwie klasy i ze etykieta nr 0 odpowiada klasie -1
        self.w_ = np.zeros(n + 1)
        self.k_ = 0
        XX = np.c_[np.ones(m), X]

        while True:
            E = [] # Lista na indeksy przykladow zle sklasyfikowanych
            for i in range(m):
                s = self.w_.dot(XX[i])
                y_pred = 1 if s > 0.0 else -1
                if y_pred != yy[i]:
                    E.append(i)
            if len(E) == 0 or self.k_ > self.kmax_:
                return
            i = E[np.random.randint(0, len(E))]
            self.w_ = self.w_ + self.learning_rate_* yy[i] * XX[i]
            self.k_ += 1

    def predict(self, X):        
        sums = self.decision_function(X)
        m = X.shape[0]
        predictions = np.empty(m, dtype=self.class_labels_.dtype)
        predictions[sums <= 0] = self.class_labels_[0]
        predictions[sums > 0] = self.class_labels_[1]
        return predictions
    
    def decision_function(self, X):
        m = X.shape[0]
        return self.w_.dot((np.c_[np.ones(m), X]).T)

    