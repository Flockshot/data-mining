from apriori_python import apriori

itemSetList = []
file_path = 'dataset/Apple_sequence_dataset.txt'
file1 = open(file_path, "r")
lines = file1.readlines()

i = 0
for line in lines:
    itemSetList.append([])
    for character in line:
        if character.isalpha():
            itemSetList[i].append(character)
    itemSetList[i] = list(set(itemSetList[i]))
    i += 1

#print(itemSetList)
file1.close()


freqItemSet, rules = apriori(itemSetList, minSup=0.75, minConf=0.75)
print(freqItemSet)
#print(rules)


all_Items = []
total_freq_items = 0
for key in freqItemSet:
    freqItem = freqItemSet[key]
    all_Items.append(freqItem)
    print(freqItem)
    print("Length", len(freqItem))
    total_freq_items += len(freqItem)
print("Total freq items", total_freq_items)

