"""env(센서) 원본 CSV를 엑셀(.xlsx)로 그대로 변환.

실행 전 필요: pip install openpyxl
"""
import os

import pandas as pd

OUT_PATH = "eda_outputs/env_data.xlsx"

os.makedirs("eda_outputs", exist_ok=True)

train_X = pd.read_csv("dataset/train/env/train_X.csv")
train_y = pd.read_csv("dataset/train/env/train_y.csv")
test_X = pd.read_csv("dataset/test/env/test_X.csv")

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    train_X.to_excel(writer, sheet_name="train_X", index=False)
    train_y.to_excel(writer, sheet_name="train_y", index=False)
    test_X.to_excel(writer, sheet_name="test_X", index=False)

print(f"저장 완료: {OUT_PATH}")
print("시트 목록: train_X, train_y, test_X")
