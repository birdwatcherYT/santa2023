import pandas as pd

base_csv = "santa-2023/sample_submission.csv"
txt_dir = "output"

sample_submission = pd.read_csv(base_csv)
case_num = 398

submit = sample_submission.copy()
for i in range(case_num):
    filename = f"{txt_dir}/{i}.txt"
    try:
        with open(filename) as f:
            ans = f.readline()
            if len(ans.split(".")) <= len(submit.at[i, "moves"].split(".")):
                submit.at[i, "moves"] = ans
    except:
        print(f"no exist: {i}")

submit.to_csv("submission.csv", index=False)
score = submit["moves"].str.split(".").map(len).sum()
print(score)
