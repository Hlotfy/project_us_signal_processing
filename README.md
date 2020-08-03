nn_eda_model.pkl

This is the neural network model used to predict arousal in the eda_processing_script.py3. It has 21 hidden layers with 16 neurons each:
'''clf = MLPClassifier(solver='lbfgs',activation='tanh',learning_rate='invscaling', hidden_layer_sizes=(16, 21), random_state=12)
MLPClassifier(solver='lbfgs',activation='tanh',learning_rate='invscaling', hidden_layer_sizes=(16, 21), random_state=12)'''


eda_scaler.pkl

This is the sklearn scaler object that is fitted to perform feature scaling according to the data used to train the neural network.


 

