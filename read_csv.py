import sys,csv
filename=sys.argv[1] #"/data0/congmin/data/query_text_9_days.sel.tag.csv"
with open(filename, 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.DictReader(f, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
            if not row:
                continue
            print('\t'.join([ row[key] for key in row.keys()]))
            #print(row['keywords'], row['text'])
