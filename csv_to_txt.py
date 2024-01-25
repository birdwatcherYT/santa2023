import pandas as pd

# submission = "santa-2023/sample_submission.csv"
submission = "submission.csv"
txt_dir = "output"

submit = pd.read_csv(submission)
for i, row in submit.iterrows():
    filename = f"{txt_dir}/{row['id']}.txt"
    with open(filename, "w") as f:
        f.write(row["moves"])
