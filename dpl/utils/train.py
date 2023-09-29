import csv
import os
import signal
import time
from typing import Union

from deepproblog.dataset import DataLoader
from deepproblog.model import Model
from deepproblog.train import TrainObject
from deepproblog.utils.logger import Logger
from deepproblog.utils.stop_condition import EpochStop
from deepproblog.utils.stop_condition import StopCondition

from dpl.utils.confusion_matrix import ConfusionMatrix
from dpl.utils.evaluate import get_confusion_matrix
from dpl.utils.logger import DataLogger


class TrainModel(TrainObject):
    def train(
            self,
            loader: DataLoader,
            stop_criterion: Union[int, StopCondition],
            verbose: int = 1,
            loss_function_name: str = "cross_entropy",
            with_negatives: bool = False,
            log_iter: int = 100,
            initial_test: bool = True,
            **kwargs
    ) -> Logger:

        self.previous_handler = signal.getsignal(signal.SIGINT)
        loss_function = getattr(self.model.solver.semiring, loss_function_name)

        self.data_logger = DataLogger(r'log/', 'NeSy')
        self.batch_size = loader.batch_size
        self.accumulated_loss = 0
        self.timing = [0, 0, 0]
        self.epoch = 0
        self.start = time.time()
        self.prev_iter_time = time.time()
        epoch_size = len(loader)
        if "test" in kwargs and initial_test:
            value = kwargs["test"](self.model)
            self.logger.log_list(self.i, value)
            print("Test: ", value)

        if type(stop_criterion) is int:
            stop_criterion = EpochStop(stop_criterion)
        print("Training ", stop_criterion)

        while not (stop_criterion.is_stop(self) or self.interrupt):
            epoch_start = time.time()
            self.model.optimizer.step_epoch()
            if verbose and epoch_size > log_iter:
                print("Epoch", self.epoch + 1)
            for batch in loader:
                if self.interrupt:
                    break
                self.i += 1
                self.model.train()
                self.model.optimizer.zero_grad()
                if with_negatives:
                    loss = self.get_loss_with_negatives(batch, loss_function)
                else:
                    loss = self.get_loss(batch, loss_function)
                self.accumulated_loss += loss
                self.model.optimizer.step()
                self.log(verbose=verbose, log_iter=log_iter, **kwargs)
                for j, hook in self.hooks:
                    if self.i % j == 0:
                        hook(self)

                if stop_criterion.is_stop(self):
                    break
            if verbose and epoch_size > log_iter:
                print("Epoch time: ", time.time() - epoch_start)
            self.epoch += 1

            if 'test_set' in kwargs:
                print("Tests!")
                confusion_matrix = get_confusion_matrix(self.model, kwargs['test_set'], verbose=1,
                                                        loss_function=loss_function)
                self.log_test_result(confusion_matrix, 'nurse_entry.csv', r'experiments/sample_efficiency/log/')

        if "snapshot_name" in kwargs:
            filename = "{}_final.mdl".format(kwargs["snapshot_name"])
            print("Writing snapshot to " + filename)
            self.model.save_state(filename)

        signal.signal(signal.SIGINT, self.previous_handler)
        return self.logger

    def log_test_result(self, conf_matrix: ConfusionMatrix, file_name: str, path: str):
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isfile(path + file_name):
            with open(path + file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([('Accuracy', 'Average loss'), (conf_matrix.accuracy(), conf_matrix.avg_loss)])

        else:
            with open(path + file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow((conf_matrix.accuracy(), conf_matrix.avg_loss))

    def log_train_loss(self, iteration_loss: (int, float), file_name: str, path: str):
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isfile(path + file_name):
            with open(path + file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([('Iteration', 'Average loss'), (iteration_loss[0], iteration_loss[1])])
        else:
            with open(path + file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow((iteration_loss[0], iteration_loss[1]))


def train_model(
        model: Model,
        loader: DataLoader,
        stop_condition: Union[int, StopCondition],
        **kwargs
) -> TrainObject:
    train_object = TrainModel(model)
    train_object.train(loader, stop_condition, **kwargs)
    return train_object
