import json
import os


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    rule_1_file = 'rule_1_test_452.jsonl'
    rule_2_file = 'rule_2_test_590.jsonl'
    rule_3_file = 'rule_3_test_34.jsonl'

    rule_1_datas = load_jsonl(rule_1_file)
    rule_2_datas = load_jsonl(rule_2_file)
    rule_3_datas = load_jsonl(rule_3_file)

    test_file_list = os.listdir('model_exp_results/move_to_localhost/')
    # test_file_list = os.listdir('model_exp_results/m10/')
    for test_file in test_file_list:
        correct_count_1 = 0
        all_count_1 = 0
        correct_count_2 = 0
        all_count_2 = 0
        correct_count_3 = 0
        all_count_3 = 0

        print(f'-------------------------------{test_file}----------------------------------')
        test_file_path = f'model_exp_results/move_to_localhost/{test_file}'
        # test_file_path = f'model_exp_results/m10/{test_file}'
        predictions = load_jsonl(test_file_path)
        for predict in predictions:
            flag = 0
            for rule_1_data in rule_1_datas:
                if rule_1_data['id'] == predict['id']:
                    if predict['result'] == 1:
                        correct_count_1 += 1
                    all_count_1 += 1
                    flag = 1
                    break

            if flag == 0:
                for rule_2_data in rule_2_datas:
                    if rule_2_data['id'] == predict['id']:
                        if predict['result'] == 1:
                            correct_count_2 += 1
                        all_count_2 += 1
                        flag = 1
                        break

            if flag == 0:
                for rule_3_data in rule_3_datas:
                    if rule_3_data['id'] == predict['id']:
                        if predict['result'] == 1:
                            correct_count_3 += 1
                        all_count_3 += 1
                        flag = 1
                        break

            if flag == 0:
                print("数据有问题")

        print(f"Accuracy for 'rule1': {(correct_count_1 / all_count_1):.4f}")
        print(f"Accuracy for 'rule2': {(correct_count_2 / all_count_2):.4f}")
        print(f"Accuracy for 'rule3': {(correct_count_3 / all_count_3):.4f}")
        print(f"Accuracy for 'overall': {((correct_count_1+correct_count_2+correct_count_3) / len(predictions)):.4f}")


if __name__ == "__main__":
    main()
