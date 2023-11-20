
from preprocess import data_preprocess

def main():
    # input_answer = int(input("Would you like to predict a boardgame based on it's (1) box art or (2) quantitive attributes?\n"))
    input_answer = 2;
    if input_answer == 2:
        quant_attributes()
    else:
        raise Exception("Please use (1) or (2) to answer.")
     
def quant_attributes():
    data_preprocess(pp_type="quant")
    
if __name__ == "__main__":
    main()