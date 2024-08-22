import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    import csv
    evidence = []
    labels = []
    map_month = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}

    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            row[10] = map_month[row[10]]
            row[15] = 1 if row[15] == "Returning_Visitor" else 0
            row[16] = 1 if row[16] == "TRUE" else 0
            labels.append(1 if row[-1] == "TRUE" else 0)
            evidence.append([int(row[i]) if i in [0, 2, 4, 11, 12, 13, 14, 15, 16] else float(row[i]) for i in range(len(row)-1)])

    return evidence, labels



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives=sum(1 for true, pred in zip(labels, predictions) if true==1 and pred==1)
    true_negatives=sum(1 for true, pred in zip(labels, predictions) if true==0 and pred==0)
    sensitivity= true_positives/sum(labels)
    specificity = true_negatives/(len(labels)-sum(labels))
    return sensitivity, specificity

if __name__ == "__main__":
    main()
