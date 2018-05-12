import csv


def readFile(dir="data/", csv_file="driving_log.csv", fieldNames=("center","left","right","steering")):
    """Read data in the fieldNames from the csv_file (in directory dir)."""

    data = []
    with open(dir + csv_file) as f:
        csvReader = csv.DictReader(f)
        for row in csvReader:
            data.append(list(row[k] for k in fieldNames))

    return data