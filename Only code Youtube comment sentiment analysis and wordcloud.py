import pandas as pd  
from textblob import TextBlob 
import matplotlib.pyplot as plt  
from wordcloud import WordCloud  
from wordcloud import ImageColorGenerator
from PIL import Image
import numpy as np

file = "mnxlxzxoZx0.xlsx"
df = pd.read_excel(file, usecols=[1, 2, 3, 4, 5]) 
print(df.head(10))


v_cmt_list = df['text'].values.tolist()
print('length of v_cmt_list is:{}'.format(len(v_cmt_list)))


score_list = []  
tag_list = [] 
for comment in v_cmt_list:
	tag = ''
	judge = TextBlob(comment)
	sentiments_score = judge.sentiment.polarity
	score_list.append(sentiments_score)
	if sentiments_score < 0:
		tag = ' negative'
	elif sentiments_score == 0:
		tag = ' neutral'
	else:
		tag = ' positive'
	tag_list.append(tag)
df['sentiment score'] = score_list
df['results'] = tag_list
df.to_excel(' sentiment analysis results.xlsx', index=None)

print(df.head(10))

print(df.groupby(by=['results']).count()['text'])

v_cmt_str = ".".join(v_cmt_list)

stopwords = ['the', 'a', 'and', 'of', 'it', 'her', 'she', 'if', 'I', 'is', 'not', 'your', 'there', 'this',
             'that', 'to', 'you', 'in', 'as', 'for', 'are', 'so', 'was', 'but', 'with', 'they', 'have']
coloring = np.array(Image.open("vedio screenshots.jpeg"))
backgroud_Image = coloring  
wc = WordCloud(
	scale=3,  
	background_color="white", 
	max_words=1000, 
	font_path='/System/Library/Fonts/SimHei.ttf',  
	stopwords=stopwords, 
	mask=backgroud_Image, 
	color_func=ImageColorGenerator(coloring),  
	max_font_size=100, 
	random_state=240 
)
wc.generate(v_cmt_str) 
wc.to_file('Wordcloud graph.png') 
# display(Image.open('vedio screenshots.jpeg')) 
# wc.to_image() 
