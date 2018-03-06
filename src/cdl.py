import ast

CDL_FN = '../data/CDL_crops.txt'
with open(CDL_FN, 'r') as file:
    for line in file:
        break

CDL_NAMES = ast.literal_eval(line)
CDL_LABELS = dict((v,k) for k,v in CDL_NAMES.items())