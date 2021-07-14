import pickle
with open('trainHistoryDict.txt', 'rb') as f:
    history = pickle.load(f)
    print(history)