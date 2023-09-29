import csv
import os
from typing import Dict, Any, List


class DataLogger:
    def __init__(self, path: str, model_type: str):
        self.__path = path
        self.__model_type = model_type
        self.log_data: Dict[str, Dict[int, Any]] = dict()
        self.keys: List[str] = list()
        self.indices: List[int] = list()

    def log(self, key: str, indice: int, value: Any):
        if key not in self.keys:
            self.log_data[key] = dict()
            self.keys.append(key)
        self.log_data[key][indice] = value

    def write_to_csv(self, name: str, header: List[str]):
        data: Dict[int, Any] = self.log_data[name]
        with open(self.__path + name + '_' + self.__model_type + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for key, value in data.items():
                writer.writerow([key, value])


if __name__ == '__main__':
    dic = {1: 2, 2: 3, 3: 4}
    print(dic.items())
    with open('test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Key', 'Value'])
        for key, value in dic.items():
            print('Key: '+str(key)+' Value:'+str(value))
            writer.writerow([key, value])
