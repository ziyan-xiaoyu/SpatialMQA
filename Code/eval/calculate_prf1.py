import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")

all_correct = 0


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line.strip())
            if tmp['result'] == 1:
                tmp['output'] = tmp['answer']
            data.append(tmp)
    return data


def main():
    file_list = os.listdir("./model_exp_results/calculate_PRF1/")
    # file_list = os.listdir("./human_results_en/")

    # for test_file_path_item in file_list:
    #     test_file_path = "./model_exp_results/" + test_file_path_item
    #
    #     data = load_jsonl(test_file_path)
    #     predictions = []
    #     for item in data:
    #         predictions.append(item['result'])
    #     answers = []
    #     for item in data:
    #         answers.append(item['answer'])
    #
    #     accuracy = accuracy_score(answers, predictions)
    #
    #     # 计算精确率
    #     precision = precision_score(answers, predictions, average='weighted')
    #
    #     # 计算召回率
    #     recall = recall_score(answers, predictions, average='weighted')
    #
    #     # 计算F1值
    #     f1 = f1_score(answers, predictions, average='weighted')
    #     print("------------------------------------" + test_file_path_item + "-----------------------------------")
    #     accuracy = "{:,.2f}".format(accuracy)
    #     precision = "{:,.2f}".format(precision)
    #     recall = "{:,.2f}".format(recall)
    #     f1 = "{:,.2f}".format(f1)
    #
    #     print(f"accuracy: {accuracy}")
    #     print(f"Precision: {precision}")
    #     print(f"Recall: {recall}")
    #     print(f"F1 Score: {f1}")

    for test_file_path_item in file_list:
        test_file_path = "./model_exp_results/calculate_PRF1/" + test_file_path_item
        # test_file_path = "./human_results_en/" + test_file_path_item

        print("------------------------------------" + test_file_path_item + "-----------------------------------")

        data_list = load_jsonl(test_file_path)
        print(len(data_list))
        on_pre_num, below_pre_num, left_pre_num, right_pre_num, front_pre_num, behind_pre_num, \
            on_label_num, below_label_num, left_label_num, right_label_num, front_label_num, behind_label_num, \
            on_true_num, below_true_num, left_true_num, right_true_num, front_true_num, behind_true_num = \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for data in data_list:
            if data['output'] in "on/above":
                on_pre_num += 1
            elif data['output'] in "below":
                below_pre_num += 1
            elif data['output'] in "left of":
                left_pre_num += 1
            elif data['output'] in "right of":
                right_pre_num += 1
            elif data['output'] in "in front of":
                front_pre_num += 1
            elif data['output'] in "behind":
                behind_pre_num += 1

            if data['answer'] == "on/above":
                on_label_num += 1
            elif data['answer'] == "below":
                below_label_num += 1
            elif data['answer'] == "left of":
                left_label_num += 1
            elif data['answer'] == "right of":
                right_label_num += 1
            elif data['answer'] == "in front of":
                front_label_num += 1
            elif data['answer'] == "behind":
                behind_label_num += 1

            if data['answer'] == data['output']:
                if data['answer'] == "on/above":
                    on_true_num += 1
                elif data['answer'] == "below":
                    below_true_num += 1
                elif data['answer'] == "left of":
                    left_true_num += 1
                elif data['answer'] == "right of":
                    right_true_num += 1
                elif data['answer'] == "in front of":
                    front_true_num += 1
                elif data['answer'] == "behind":
                    behind_true_num += 1
        if on_pre_num == 0:
            on_pre_num = 1
        if below_pre_num == 0:
            below_pre_num = 1
        if left_pre_num == 0:
            left_pre_num = 1
        if right_pre_num == 0:
            right_pre_num = 1
        if front_pre_num == 0:
            front_pre_num = 1
        if behind_pre_num == 0:
            behind_pre_num = 1

        p = (on_true_num / on_pre_num + below_true_num / below_pre_num + left_true_num / left_pre_num
             + right_true_num / right_pre_num + front_true_num / front_pre_num
             + behind_true_num / behind_pre_num) / 6
        print(on_true_num, on_pre_num, below_true_num, below_pre_num, left_true_num, left_pre_num,
              right_true_num, right_pre_num, front_true_num, front_pre_num,
              behind_true_num, behind_pre_num)
        r = (on_true_num / on_label_num + below_true_num / below_label_num + left_true_num / left_label_num
             + right_true_num / right_label_num + front_true_num / front_label_num +
             behind_true_num / behind_label_num) / 6
        f1 = 2 * p * r / (p + r)
        acc = (on_true_num + below_true_num + left_true_num + right_true_num + front_true_num +
               behind_true_num) / (on_label_num + below_label_num + left_label_num + right_label_num
                                   + front_label_num + behind_label_num)

        # p = ((on_true_num + below_true_num) / (on_pre_num + below_pre_num) + (left_true_num + right_true_num)
        #      / (left_pre_num + right_pre_num) + (front_true_num + behind_true_num) / (front_pre_num +
        #                                                                               behind_pre_num)) / 3
        # r = ((on_true_num + below_true_num) / (on_label_num + below_label_num) + (left_true_num + right_true_num)
        #      / (left_label_num + right_label_num) + (front_true_num + behind_true_num) / (front_label_num +
        #                                                                                   behind_label_num)) / 3
        # f1 = 2 * p * r / (p + r)
        # acc = (on_true_num + below_true_num + left_true_num + right_true_num + front_true_num +
        #        behind_true_num) / (on_label_num + below_label_num + left_label_num + right_label_num
        #                            + front_label_num + behind_label_num)

        print(f"accuracy: {acc}")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"F1 Score: {f1}")


if __name__ == "__main__":
    main()
