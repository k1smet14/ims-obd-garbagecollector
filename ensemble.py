import numpy as np
import pandas as pd
import os
from collections import Counter

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# # soft voting
# if __name__ == '__main__':
#     create_dir('./prediction/ensemble')
#     soft1path = '/opt/ml/MyBaseline/prediction/xlm_setting01/checkpoint-1800_soft.csv'
#     soft2path = '/opt/ml/MyBaseline/prediction/original_setting_using_allx2data/checkpoint-10000_soft.csv'
#     save_name = 'xlm_setting01+original_setting_using_allx2data'

#     soft1 = np.array(pd.read_csv(soft1path, delimiter=','))
#     soft2 = np.array(pd.read_csv(soft2path, delimiter=','))

#     sum_soft = soft1+soft2
#     sum_soft = np.argmax(sum_soft,axis=1)

#     output = pd.DataFrame(sum_soft, columns=['pred'])
#     output.to_csv(f"./prediction/ensemble/{save_name}.csv", index=False)


# hard voting (but class are different -> class = 0)
if __name__ == '__main__':
    create_dir('./prediction/ensemble')
    hard1path = '/opt/ml/MyBaseline/prediction/xlm_setting01/checkpoint-1800.csv'
    hard2path = '/opt/ml/MyBaseline/prediction/original_setting_using_allx2data/checkpoint-10000.csv'
    hard3path = '/opt/ml/MyBaseline/prediction/ensemble/original_setting_alltrain+original_setting_using_allx2data.csv'
    save_name = 'original_setting_alltrain+original_setting_using_allx2data+ensemble(hard)'

    hard1 = np.array(pd.read_csv(hard1path, delimiter=',')).flatten()
    hard2 = np.array(pd.read_csv(hard2path, delimiter=',')).flatten()
    hard3 = np.array(pd.read_csv(hard3path, delimiter=',')).flatten()


    output = []
    for a, b, c in zip(hard1, hard2, hard3):
        if a != b and a != c and b != c:
            output.append(0)
        else:
            cnt_labels = list(Counter([a, b, c]).items())
           
            max_label = max(cnt_labels, key=lambda x: x[1])[0]
            output.append(max_label)
        print(a, b, c, output[-1])

    output = pd.DataFrame(output, columns=['pred'])
    output.to_csv(f"./prediction/ensemble/{save_name}.csv", index=False)