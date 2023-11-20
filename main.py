from bgg_rating import bgg_rating
from preprocess import quant_data_preprocess
from preprocessCNN import cnn_data_preprocess
from quant_rating import quant_models

def main():
    input_answer = int(input("Would you like to predict a boardgame based on it's (1) box art or (2) quantitive attributes?\n"))
    
    if input_answer == 1:
        cnn_data_preprocess()
        bgg_rating()
    elif input_answer == 2:
        X_train, X_test, y_train, y_test = quant_data_preprocess()
        quant_models(X_train, X_test, y_train, y_test)
    else:
        raise Exception("Please use (1) or (2) to answer.")
    
if __name__ == "__main__":
    main()