"""온라인테스트1 추론 (샘플): output/submission.csv 생성 (형식은 train_y.csv 와 동일)."""
import csv
import os
import pickle

COLUMNS = ["timestamp", "soil_moisture", "soil_ec", "soil_temp"]


def main():
    with open("model/model.pkl", "rb") as f:
        pickle.load(f)
    os.makedirs("output", exist_ok=True)
    with open("output/submission.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(COLUMNS)
    print("hello world: output/submission.csv saved")


if __name__ == "__main__":
    main()
