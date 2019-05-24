import json

with open('dataset/tacred/dev.json') as file:
    dev = json.load(file)

positives = []
negatives = []

for d in dev:
    if d['relation'] == "no_relation":
        negatives.append((' '.join(d['token']), ' '.join(d['token'][d['subj_start']:d['subj_end']+1]), ' '.join(d['token'][d['obj_start']:d['obj_end']+1]), d['relation']))
    else:
        positives.append((' '.join(d['token']), ' '.join(d['token'][d['subj_start']:d['subj_end']+1]),
                          ' '.join(d['token'][d['obj_start']:d['obj_end']+1]), d['relation']))

with open('analysis/compare.txt', 'w') as file:
    for p in positives[:10]:
        file.write(p[0]+" :::: " + p[1] + " ::::: "+p[2] + " :::: "+ p[3])
        file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    for n in negatives[:10]:
        file.write(n[0] + " :::: " + n[1] + " ::::: " + n[2] + " :::: "+ p[3])
        file.write("\n")
