import matplotlib.pyplot as plt

with open('../s', 'r') as file:
    content = file.readlines()
    losses = [float(line.split(' ')[1]) for line in content]
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()