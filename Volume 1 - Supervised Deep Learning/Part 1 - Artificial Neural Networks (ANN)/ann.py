# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Encode Geography
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Encode Genre
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Part 2 - Build ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Build ANN
def build_classifier():
    classifier = Sequential()
    
    # Ajout couche d'entrée & couche cachée
    #units=6 : 6 neuronnes (moyenne entre entré et sortie, 11 + 1)
    #activation=relu : Fonction d'activation redresseur
    #input_dim=11 : Taille de la couche d'entrée. A faire uniquement lors de l'init de la premiere couche cachée
    classifier.add(Dense(units=6, 
                         activation="relu", 
                         kernel_initializer='uniform',
                         input_dim=11))
    classifier.add(Dropout(rate=0.1)) # Retirera 10% des neuronnes a chaque itération. Permet d'améliorer la fiabilité en évitant une trop grand dépendence entre les neuronnes     
    
    #Ajout 2eme couche cachée
    classifier.add(Dense(units=6, 
                         activation="relu", 
                         kernel_initializer='uniform')) 
    classifier.add(Dropout(rate=0.1)) # Retirera 10% des neuronnes a chaque itération. Permet d'améliorer la fiabilité en évitant une trop grand dépendence entre les neuronnes     
        
    
    #Ajout couche Sortie
    #units=1 : 1 seul valeur en sortie (le client quitte ou non la banque)
    #activation=sigmoid : Fonction sigmoide : Permet d'avoir une propabilité 
    classifier.add(Dense(units=1,  
                         activation="sigmoid", 
                         kernel_initializer='uniform')) 
    
    
    #Compilation de l'ANN
    #optimizer="adam" : Utilisation de l'algorythme du gradient stochastique
    #loss="binary_crossentropy" : Fonction de cout logarythmique
    #metrics=['accuracy']: Mesure de la performance du model
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    return classifier


#Entrainement de l'ANN
#batch_size=10 : Taille du lot d'observation (apprentissage par lot)
#epochs=100 : Nombre d'époque => Nbr de fois que l'on rejoue l'apprentissage
classifier = build_classifier()
classifier.fit(X_train, y_train, batch_size=25, epochs=500)
# loss = Cout de la fonction
# acc = Précision de la prédiction

# Pour persister un classifier : https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# PREDICTION DU JEU DE TEST
y_pred = classifier.predict(X_test) #Retourne la probabilité de partir pour chaque clients
y_pred = (y_pred > 0.5) #Convertit en boolean les clients qui ont une proba de partir > a 50%

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





# PREDICTION D'UN CAS 
# client = np.array([[0.0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])
# client = sc.transform(client)
# predict = classifier.predict(client) #Param de sortie

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)






#PARTIE 4 : Recherche des meilleurs paramètre pour les neuronnes

#import keras
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#
## Build ANN
#def build_classifier(optimizer):
#    classifier = Sequential()
#    
#    # Ajout couche d'entrée & couche cachée
#    #units=6 : 6 neuronnes (moyenne entre entré et sortie, 11 + 1)
#    #activation=relu : Fonction d'activation redresseur
#    #input_dim=11 : Taille de la couche d'entrée. A faire uniquement lors de l'init de la premiere couche cachée
#    classifier.add(Dense(units=6, 
#                         activation="relu", 
#                         kernel_initializer='uniform',
#                         input_dim=11))
#    classifier.add(Dropout(rate=0.1)) # Retirera 10% des neuronnes a chaque itération. Permet d'améliorer la fiabilité en évitant une trop grand dépendence entre les neuronnes     
#    
#    #Ajout 2eme couche cachée
#    classifier.add(Dense(units=6, 
#                         activation="relu", 
#                         kernel_initializer='uniform')) 
#    classifier.add(Dropout(rate=0.1)) # Retirera 10% des neuronnes a chaque itération. Permet d'améliorer la fiabilité en évitant une trop grand dépendence entre les neuronnes     
#        
#    
#    #Ajout couche Sortie
#    #units=1 : 1 seul valeur en sortie (le client quitte ou non la banque)
#    #activation=sigmoid : Fonction sigmoide : Permet d'avoir une propabilité 
#    classifier.add(Dense(units=1,  
#                         activation="sigmoid", 
#                         kernel_initializer='uniform')) 
#    
#    
#    #Compilation de l'ANN
#    #optimizer="adam" : Utilisation de l'algorythme du gradient stochastique
#    #loss="binary_crossentropy" : Fonction de cout logarythmique
#    #metrics=['accuracy']: Mesure de la performance du model
#    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
#
#    return classifier
#
#
#classifier = KerasClassifier(build_fn=build_classifier)
#parameters = {
#        "batch_size": [25, 32], 
#        "epochs": [100, 500],
#        "optimizer": ["adam", "rmsprop"]
#    }
#
## Va executer un apprentissage pour chaque params, et retournera les meilleurs params a utiliser.
##Entrainement x10 pour optimiser la precisions
##param_grid=parameters: Map de params a utiliser pour exectuer avec chaque valeur l'apprentissage
##cv=10 : Nombre d'itération de l'apprentissage
##n_jobs=-1 : Nombre de coeur utilisé pour la parralélisation
#grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=-1)
#grid_search = grid_search.fit(X_train, y_train) # Exec
#best_params = grid_search.best_params_ # doit retourner {optimizer: 'adam', epochs: 500, batch_size: 25}
#best_precision = grid_search.best_score_ # doit retourner env. 0.85













