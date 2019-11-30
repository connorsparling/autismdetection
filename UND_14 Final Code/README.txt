There are three main programs that we have:

1) Backpropogation Neural Network using Keras: "forward.py" and "preprocess.py"
Requirements: (sklearn, numpy, pandas, matplotlib, tensorflow, keras)
This program can be run with many different flags but the main output is run by calling the command "python3 forward.py -d -p -i -s 0 -t 100"
-d represents the default csv data file location
-p represents showing the final accuracy and weight plots
-i calculates and prints the importance values
-s <num 0-3> represents which subset of the data to use from preprocess.py
-t <num> represents the number of epochs to train for

2) Multi Layer Perceptron: "MLP.py"
Requirements: (sklearn, numpy, pandas, matplotlib)
This program can be run by calling "python3 MLP.py"

3) Other ML Algorithms: "Variety_ML_Algos.py"
Requirements: (sklearn, numpy, pandas, matplotlib)
This program can be run by calling "python3 Variety_ML_Algos.py"