import json
import os

all_correct = 0
position = {'y': ["on/above", "below"], 'z': ["in front of", "behind"],
            'x': ["left of", "right of"]}


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def change_output(predict):
    if "front" in predict['output']:
        predict['output'] = "in front of"
    elif "behind" in predict['output']:
        predict['output'] = "behind"
    elif "left" in predict['output']:
        predict['output'] = "left of"
    elif "right" in predict['output']:
        predict['output'] = "right of"
    elif "below" in predict['output']:
        predict['output'] = "below"
    elif "on" in predict['output'] or "above" in predict['output']:
        predict['output'] = "on/above"
    else:
        predict['output'] = "--"
    return predict['output']


def calculate_accuracy(predictions, option):
    correct_count = 0
    total_count = 0
    option1 = position[option][0]
    option2 = position[option][1]
    for predict in predictions:
        if predict['answer'] == option1 or predict['answer'] == option2:
            total_count += 1
            if predict['result'] == 1:
                correct_count += 1
    global all_correct
    all_correct += correct_count
    if total_count == 0:
        return 0.0
    print(option + ": " + str(correct_count) + "; " + str(total_count))
    return correct_count / total_count


def main():
    # test_file_list = os.listdir('model_exp_results/m10/')
    test_file_list = os.listdir('model_exp_results/move_to_localhost/')
    for test_file in test_file_list:
        print(f'-------------------------------{test_file}----------------------------------')
        test_file_path = f'model_exp_results/move_to_localhost/{test_file}'
        # test_file_path = f'model_exp_results/m10/{test_file}'

        # 重新处理一下on/above的输出结果：
        predictions = load_jsonl(test_file_path)
        change_count = 0
        # in_front_of_count = 0
        for predict in predictions:
            if 501 > predict['id'] > 0:
                if "on" in predict['output'] or "above" in predict['output']:
                    predict['result'] = 1
                    change_count += 1
            # elif 1756 > predict['id'] > 945:  # 处理in front of(id:958~1751)
            #     if "front" in predict['output'] or "front" in predict['answer']:
            #         in_front_of_count += 1
        # for predict in predictions:
        #     if 512 > predict['id']:     # 处理on/above(id:1~500)
        #         if "on" in predict['output'] or "above" in predict['output']:
        #             predict['result'] = 1
        #             predict['output'] = "on/above"
        #             change_count += 1
        #         else:
        #             predict['output'] = change_output(predict)
        #     elif 958 > predict['id'] > 500:  # 处理below(id:512~945)
        #         if "below" in predict['output']:
        #             predict['result'] = 1
        #             predict['output'] = "below"
        #             change_count += 1
        #         else:
        #             predict['output'] = change_output(predict)
        #     elif 1756 > predict['id'] > 945:  # 处理in front of(id:958~1751)
        #         if "front" in predict['output']:
        #             predict['result'] = 1
        #             predict['output'] = "in front of"
        #             change_count += 1
        #         else:
        #             predict['output'] = change_output(predict)
        #     elif 2513 > predict['id'] > 1751:  # 处理behind(id:1756~2509)
        #         if "behind" in predict['output']:
        #             predict['result'] = 1
        #             predict['output'] = "behind"
        #             change_count += 1
        #         else:
        #             predict['output'] = change_output(predict)
        #     elif 3910 > predict['id'] > 2509:  # 处理left of(id:2513~3906)
        #         if "left" in predict['output']:
        #             predict['result'] = 1
        #             predict['output'] = "left of"
        #             change_count += 1
        #         else:
        #             predict['output'] = change_output(predict)
        #     elif predict['id'] > 3906:  # 处理right of(id:3910~5391)
        #         if "right" in predict['output']:
        #             predict['result'] = 1
        #             predict['output'] = "right of"
        #             change_count += 1
        #         else:
        #             predict['output'] = change_output(predict)

        # with open(test_file_path, 'w', encoding='utf-8') as f:
        #     for predict in predictions:
        #         f.write(json.dumps(predict) + '\n')
        # print(f"修改此文件中的on/above的结果数据共：{change_count}条")
        # print(f"此文件中的in front of的结果数据共：{in_front_of_count}条")

        predictions = load_jsonl(test_file_path)
        print(len(predictions))
        # answers = load_jsonl('../SpatialMQA(En)/all_en_test_1076_sort.jsonl')

        accuracies = {}
        for option in ['x', 'y', 'z']:
            accuracies[option] = calculate_accuracy(predictions, option)

        global all_correct
        for option, accuracy in accuracies.items():
            print(f"Accuracy for '{option}': {accuracy:.4f}")
        print(f"Accuracy for 'overall': {(all_correct / len(predictions)):.4f}")
        all_correct = 0


if __name__ == "__main__":
    main()
