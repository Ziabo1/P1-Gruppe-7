import randomforestbest
import knn
import getuserimput

def main():
    randomforestbest.TrainAndSaveModel()
    getuserimput.predict_stress()

if __name__ == "__main__":
    main()