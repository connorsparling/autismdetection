import numpy as np
import csv
from sklearn.model_selection import train_test_split

# default values
#FILENAME = "autism-screening-for-toddlers/Toddler Autism dataset July 2018.csv"
FILENAME = "Toddler Autism.csv"
#ALL_COLUMNS = ["Case_No", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age_Mons", "Qchat-10-Score", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who completed the test", "Class/ASD Traits"] 
RELEVANT_COLUMNS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age_Mons", "Qchat-10-Score", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]
COLUMN_TYPE = ["BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "NORM", "NORM", "ONEH", "ONEH", "YORN", "YORN"]
# RELEVANT_COLUMNS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
# COLUMN_TYPE = ["BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL", "BOOL"]
CLASSIFIER_COLUMN = "Class/ASD Traits"

def get_one_hot(data, header):
    types = []
    zeros = []
    columns = []
    count = 0
    for d in data:
        x = "{}Is{}".format(header, d.title().replace(" ",""))
        count += 1
        if x not in types:
            types.append(x)
            columns.append(zeros.copy())
        for i in range(len(columns)):
            if x == types[i]:
                columns[i].append(float(1.0))
                zeros.append(float(0.0))
            else:
                columns[i].append(float(0.0))
    return np.array(columns), types

def get_yes_no(data):
    new = []
    for x in data:
        if x.lower() == "yes":
            new.append(float(1.0))
        else:
            new.append(float(0.0))
    return np.array(new)

def get_normalized(data):
    new = []
    d2 = []
    # this should not be neccessary but the max function is 
    #    failing for the data array and I give up for now
    for d in data:
        d2.append(float(d))
    m = np.max(d2)
    for d in d2:
        new.append(d / m)
    return np.array(new)

def get_bool(data):
    new = []
    for x in data:
        if x == 1:
            new.append(float(1.0))
        else:
            new.append(float(0.0))
    return np.array(new)

def load_data():
    X = []
    y = []
    headers = []
    with open(FILENAME) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers_file = next(csv_reader)
        y_index = headers_file.index(CLASSIFIER_COLUMN)
        headers = [h for h in headers_file if h in RELEVANT_COLUMNS]
        for row in csv_reader:
            dp = [row[i] for i in range(len(headers_file)) if headers_file[i] in RELEVANT_COLUMNS]
            X.append(dp)
            y.append(row[y_index])
    csv_file.close()

    X_T = np.transpose(X)
    X_New_T = []
    H = []
    for i in range(len(headers)):
        col = None
        if COLUMN_TYPE[i] is "BOOL":
            X_New_T.append(get_bool(X_T[i]))
            H.append(headers[i])
        elif COLUMN_TYPE[i] is "NORM":
            X_New_T.append(get_normalized(X_T[i]))
            H.append(headers[i])
        elif COLUMN_TYPE[i] is "YORN":
            X_New_T.append(get_yes_no(X_T[i]))
            H.append(headers[i])
        elif COLUMN_TYPE[i] is "ONEH":
            oneH, head = get_one_hot(X_T[i], headers[i])
            for j in range(len(head)):
                col = oneH[j]
                X_New_T.append(col)
                H.append(head[j])
    X = np.transpose(X_New_T)

    y = get_yes_no(y)

    print("FINAL HEADERS: {}".format(len(H)))
    print(H)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
    print("\nTRAIN: {}\tTEST: {}".format(X_train.shape, X_test.shape))

    return X_train, X_test, y_train, y_test, H

def main():
    X_train, X_test, y_train, y_test = load_data()

    # with open('X_train.csv', 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file, delimiter=',')
    #     csv_writer.writerow(RELEVANT_COLUMNS)
    #     for row in X_test:
    #         csv_writer.writerow(row)
    # csv_file.close()

if __name__ == '__main__':
	main()