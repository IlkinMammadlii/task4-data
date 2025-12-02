from load_clean import load_and_clean

df1 = load_and_clean(r"C:\Uni\internship\task4\DATA1")
df2 = load_and_clean(r"C:\Uni\internship\task4\DATA2")
df3 = load_and_clean(r"C:\Uni\internship\task4\DATA3")

print(len(df1), len(df2), len(df3))
