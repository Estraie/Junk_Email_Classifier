# -*- coding: utf-8 -*-
import jieba

# word = "秦始皇派蒙恬还原神舟十二号飞船"
word = "原神舟十二号飞船"
ct = jieba.cut(word)
words = []
for w in ct:
		words.append(w)
		
print(words)