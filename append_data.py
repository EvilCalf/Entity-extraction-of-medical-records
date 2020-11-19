import os

txt = "1.腰椎退行性变。2.L5向前稍滑脱。3.T11压缩骨折。4.心肺未见明显异常。"
char_list = list(txt)
with open('data\\train.txt', encoding="utf-8", mode="a") as f:
    for i, char in enumerate(char_list):
        f.write(char)
        f.write("\n")
