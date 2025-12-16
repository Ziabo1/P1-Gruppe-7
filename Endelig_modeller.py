import randomforestbest
import knn
def main():
    randomforestbest.TrainAndSaveModel()
    import getuserimput
    getuserimput.predict_stress()

if __name__ == "__main__":
    main()