from utils.io_utils import get_predictions_from_file, save_results_to_file
from utils.parser import get_given_parameters_parser
from utils.scorer import get_scorer

import numpy as np
import pandas as pd

def main(args):
    print("Evaluate model " + args.model_name)

    predictions = get_predictions_from_file(args)
    scorer = get_scorer(args)

    for pred in predictions:
        # [:,0] is the truth and [:,1:] are the prediction probabilities

        truth = pred[:, 0]
        out = pred[:, 1:]
        pred_label = np.argmax(out, axis=1)

        scorer.eval(truth, pred_label, out)

    result = scorer.get_results()
    print(result)
    # /////////////////////////////////////////////////////////
    file_path = "/content/"
    if os.path.isfile(file_path):
        # Đọc dữ liệu từ file pandas đã tồn tại
        df = pd.read_csv(file_path)
        
        # Thêm dữ liệu mới vào dataframe đã tồn tại
        new_data = pd.DataFrame.from_dict(result, orient='index', columns=["Value"])
        new_data.insert(0, "Name", args.dataset)
        new_data.insert(1, "Model", args.model_name)
        df = pd.concat([df, new_data], ignore_index=True)
        
        # Lưu dữ liệu mới vào file pandas
        df.to_csv(file_path, index=False)
    else:
        print("Create new report!")
        # Tạo dataframe mới và lưu vào file pandas
        df = pd.DataFrame.from_dict(result, orient='index', columns=["Value"])
        df.insert(0, "Name", args.dataset)
        df.insert(1, "Model", args.model_name)
        df.to_csv(file_path, index=False)
    # ////////////////////////////////////////////////////////
    save_results_to_file(args, result)


if __name__ == "__main__":

    # Also load the best parameters
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
