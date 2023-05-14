import train
import pred
import conlleval

train.main()
pred.main()

path = "./output.txt"
with open(path) as f:
    file_iter = f.readlines()
conlleval.evaluate_conll_file(file_iter)
