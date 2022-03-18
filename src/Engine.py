from src.ML_Pipeline import Predict, Train_Model
from src.ML_Pipeline.Preprocess import apply
from src.ML_Pipeline.Utils import load_model, save_model
import pandas as pd
import subprocess


val = int(input(" Train -- 0 \n Predict -- 1 \n Deploy -- 2\n Enter your value :"))

if val==0:
    data = pd\
        .read_csv("../input/final_cred_data.csv",low_memory=False, index_col=0)\
        .drop_duplicates()\
        .reset_index(drop=True)

    print("Data loaded in to the pandas dataframe")

    processed_df = apply(data, is_train=True)
    ml_model, columns = Train_Model.fit(processed_df)
    model_save = save_model(ml_model, columns)
    print("Model saved in:""output/deep-ae-model")

elif val==1:
    model_path = "../output/deep-ae-model"
    ml_model, columns = load_model(model_path)
    test_data = pd\
            .read_csv("../input/test_data.csv", low_memory=False, index_col=0)\
            .drop_duplicates()\
            .reset_index(drop=True)
    proceessed_df = apply(test_data, is_train=False)
    prediction = Predict.init(proceessed_df,ml_model, columns)
    print(prediction)

else:
    # For prod deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For dev deployment
    process = subprocess.Popen(['python', 'ML_Pipeline/Deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
