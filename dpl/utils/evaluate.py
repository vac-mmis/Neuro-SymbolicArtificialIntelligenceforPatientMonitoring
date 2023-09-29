from typing import Optional, Callable

from deepproblog.dataset import Dataset
from deepproblog.model import Model
from tqdm import tqdm

from dpl.utils.confusion_matrix import ConfusionMatrix


def get_confusion_matrix(
        model: Model, dataset: Dataset, verbose: int = 0, eps: Optional[float] = None,
        loss_function: Callable = None
) -> ConfusionMatrix:
    """

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param verbose: Set the verbosity. If verbose > 0, then print confusion matrix and accuracy.
    If verbose > 1, then print all wrong answers.
    :param eps: If set, then the answer will be treated as a float, and will be considered correct if
    the difference between the predicted and ground truth value is smaller than eps.
    :return: The confusion matrix when evaluating model on dataset.
    """
    confusion_matrix = ConfusionMatrix()
    model.eval()
    total_loss = 0
    total_iterations = len(dataset)
    with tqdm(total=total_iterations, ncols=100) as pbar:
        for i, gt_query in enumerate(dataset.to_queries()):
            test_query = gt_query.variable_output()
            answer = model.solve([test_query])[0]
            actual = str(gt_query.output_values()[0])
            if len(answer.result) == 0:
                predicted = "no_answer"
                if verbose > 1:
                    print("no answer for query {}".format(gt_query))
            else:
                max_ans = max(answer.result, key=lambda x: answer.result[x])
                p = answer.result[max_ans]
                if eps is None:
                    predicted = str(max_ans.args[gt_query.output_ind[0]])
                else:
                    predicted = float(max_ans.args[gt_query.output_ind[0]])
                    actual = float(gt_query.output_values()[0])
                    if abs(actual - predicted) < eps:
                        predicted = actual
                if verbose > 1 and actual != predicted:
                    print(
                        "{} {} vs {}::{} for query {}".format(
                            i, actual, p, predicted, test_query
                        )
                    )
            confusion_matrix.add_item(predicted, actual)
            loss = loss_function(answer, gt_query.p, weight=1, q=gt_query.substitute().query)
            confusion_matrix.total_loss += loss
            pbar.update(1)

    confusion_matrix.avg_loss = confusion_matrix.total_loss / len(dataset)
    print(
        f'Test: \t Accuracy: {(100 * confusion_matrix.accuracy()):>0.1f}% \t Avg loss: {(confusion_matrix.avg_loss):>0.6f}')
    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix
