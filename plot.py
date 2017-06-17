import matplotlib.pyplot as plt

with open('losses.txt', 'r') as file:
    content = file.readlines()
    losses = [float(line) for line in content]
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

with open('accuracies.txt', 'r') as file:
    content = file.readlines()
    accuracies = [float(line) for line in content]
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
