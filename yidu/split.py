for index, line in enumerate(open('subtask1_training_part1.txt', 'r',encoding='UTF-8'), 1):
    with open('yidu/%d.json' % index, 'w+',encoding='UTF-8') as tmp:
        tmp.write(line)