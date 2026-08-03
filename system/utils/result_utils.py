import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, result_path="../results"):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, result_path)
    max_accuracy = []
    for values in test_acc:
        if len(values) > 0:
            max_accuracy.append(values.max())

    if not max_accuracy:
        print("No result files found for averaging.")
        return
    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, result_path="../results"):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        file_path = os.path.join(result_path, file_name + ".h5")
        if not os.path.exists(file_path):
            print("Missing result file:", file_path)
            continue
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False, result_path=result_path)))

    return test_acc


def read_data_then_delete(file_name, delete=False, result_path="../results"):
    file_path = os.path.join(result_path, file_name + ".h5")

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc
