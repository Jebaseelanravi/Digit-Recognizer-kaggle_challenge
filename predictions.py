pkl_filename = "pickle_model.pkl"
import pickle
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


import csv

with open('test.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i,row in enumerate(readCSV):
        predictions=pickle_model.predict([row])

        print("processed_row"+str(i+1))
        with open('Final_predictions.csv','a') as file:
            file.write(str(i+1)+','+str(predictions[0])+'\n')

