"""온라인테스트1 학습 (샘플): model/ 에 가중치 생성."""
import os
import pickle


def main():
    os.makedirs("model", exist_ok=True)
    weights = {"note": "sample hello world model"}
    with open("model/model.pkl", "wb") as f:
        pickle.dump(weights, f)
    print("hello world: model/model.pkl saved")


if __name__ == "__main__":
    main()
