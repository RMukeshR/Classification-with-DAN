## Read data

df_pos = open("Train.pos","r", encoding= "latin-1").read()
df_neg = open("Train.neg","r", encoding= "latin-1").read()
df_test = open("TestData","r", encoding= "latin-1").read()

df_pos_list = [i for i in df_pos.split("\n") if len(i) >= 2]
df_neg_list = [i for i in df_neg.split("\n") if len(i) >= 2]
df_test_list = [i for i in df_test.split("\n") if len(i) >= 2]
