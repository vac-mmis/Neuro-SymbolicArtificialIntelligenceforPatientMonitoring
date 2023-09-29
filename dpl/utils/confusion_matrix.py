from deepproblog.utils import confusion_matrix


class ConfusionMatrix(confusion_matrix.ConfusionMatrix):
    def __init__(self):
        confusion_matrix.ConfusionMatrix.__init__(self)
        self.__total_loss = 0
        self.__avg_loss = 0

    def accuracy(self):
        correct = 0
        for i in range(self.n):
            correct += self.matrix[i, i]
        total = self.matrix.sum()
        acc = correct / total
        return acc

    @property
    def avg_loss(self) -> float:
        return self.__avg_loss

    @avg_loss.setter
    def avg_loss(self, value: float):
        self.__avg_loss = value

    @property
    def total_loss(self) -> float:
        return self.__total_loss

    @total_loss.setter
    def total_loss(self, value: float):
        self.__total_loss = value
