from flask import Flask, render_template ,request
import glob
import MeCab
import urllib.request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import cv2
import time
import pickle
import os

app = Flask(__name__)


@app.route('/0615_1')
def dentaku():
  name = "電卓"
  kadai = "足し算Webアプリ（電卓）"
  return render_template('0615_1.html', title=name,description =name,kadai=kadai)

@app.route('/0615_2')
def kensaku():
  name = "検索"
  kadai = "スタッフ検索"
  return render_template('0615_2.html', title=name,description =name,kadai=kadai)

@app.route('/post-dentaku',methods=["POST"])
def calc():
  return str(eval(request.json['text']))

@app.route('/post-stuff',methods=["POST"])
def serch_stuff():
  name = request.json['text']
  f = open('./static/staff.txt', 'r')
  data = f.read().split()
  f.close()
  return str(name in data)

@app.route('/post-stuff-add',methods=["POST"])
def add_stuff():
  if request.json['pass'] == "password":
    name = request.json['text']
    f = open('./static/staff.txt', 'a')
    print(name,file=f)
    f.close()
    return "True"
  else:
    return "False"

# 第二週
@app.route('/0622')
def tfidf():
  name = "書き込み"
  kadai = "ファイル追加"
  return render_template('0622.html', title=name,description =name,kadai=kadai)

@app.route('/post-tf-text',methods=["POST"])
def add_tf():
  f = open("static/tmp/"+request.json['name']+'.txt', 'w', encoding='UTF-8')
  f.write(request.json['text'])
  f.close()
  return "True"

@app.route('/tf-culc',methods=["POST"])
def tf_culc():
  print("tf_culc start")
  files = glob.glob("./static/tmp/*.txt")
  novel = []
  for file in files:
    with open(file, "r") as afile:
      novel.append(afile.read())
  print(files)
  m = MeCab.Tagger("-Owakati")  # MeCabで分かち書きにする

  readtextlist = novel
  stringlist = ['\n'.join(u) for u in readtextlist]
  wakatilist = [parsewithelimination(u) for u in stringlist]
  wakatilist = np.array(wakatilist)

  vectorizer = TfidfVectorizer(use_idf=True, norm=None, token_pattern=u'(?u)\\b\\w+\\b')
  tfidf = vectorizer.fit_transform(wakatilist)

  tfidfpd = pd.DataFrame(tfidf.toarray())
  itemlist=sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])
  tfidfpd.columns = [u[0] for u in itemlist]  # 欄の見出し（単語）を付ける
  tfidfpd.index=[u for u in files]
  tfidfpd.to_csv("./static/output/metadata.csv")
  result = str(tfidfpd.head())
  return result

def parsewithelimination(sentense):
  slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
  slothlib_file = urllib.request.urlopen(slothlib_path)
  stopwords=[]
  for line in slothlib_file:
    ss=line.decode("utf-8").strip()
    if not ss==u'':
      stopwords.append(ss)

  elim=['数','非自立','接尾']
  part=['名詞', '動詞', '形容詞']

  m=MeCab.Tagger()
  m.parse('')
  node=m.parseToNode(sentense)
  result=''
  while node:
    if node.feature.split(',')[6] == '*': # 原形を取り出す
      term=node.surface
    else :
      term=node.feature.split(',')[6]
    if term in stopwords:
      node=node.next
      continue
    if node.feature.split(',')[1] in elim:
      node=node.next
      continue

    if node.feature.split(',')[0] in part:
      if result == '':
        result = term
      else:
        result=result.strip() + ' '+ term
    node=node.next

  return result

# === 第三週 ===

@app.route('/0629')
def calc_ruizido():
  name = "類似度計算"
  kadai = "与えられた単語の類似度計算"
  return render_template('0629.html', title=name,description =name,kadai=kadai)

@app.route('/ruizido-word',methods=["POST"])
def ruizido_word():
  df = pd.read_csv('./static/output/metadata.csv', header=0, index_col=0)
  print(df)
  qvec=np.zeros(df.columns.shape)
  keys=np.array([str(request.json['word'])])
  # keys=np.array(['10分間','鼓動'])
  for key in keys:
    if np.any(df.columns == key):
      qvec[np.where(df.columns==key)[0][0]]=1
  result=np.array([])
  for i in range(df.index.shape[0]):
    result=np.append(result, comp_sim(qvec, df.iloc[i,:].to_numpy()))
  rank=np.argsort(result)
  return_test = ""
  for index in rank[:-rank.shape[0]-1:-1]:
    return_test += '<tr><td>{}</td><td>{}</td></tr>'.format(df.index[index], result[index])
    print('{}\t{}'.format(df.index[index], result[index]))
  return return_test


# ==第四・五週==
@app.route('/0706',methods=["GET"])
def get_metadata():
  name = "メタデータ取得"
  kadai = "画像のメタデータ取得"
  return render_template('0706.html', title=name,description =name,kadai=kadai)

@app.route("/img-post",methods = ["POST"])
def img_post():
  # メタデータ読み込み
  df=pd.read_csv('./static/output/metadata_rgb.csv', header=0)
  tfilepaths=df["time"]
  rgbs=df[["R","G","B"]]


  _bytes = np.frombuffer(request.data, np.uint8)
  # decode the bytes to image directly
  img = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)
  now_time = str(time.time()).replace(".","_")
  cv2.imwrite(f'./static/tmp/{now_time}.jpg', img)
  im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  rgb_mean = np.array([np.mean(im_rgb[:,0]),np.mean(im_rgb[:,1]),np.mean(im_rgb[:,2])])
  f = open('./static/output/metadata_rgb.csv', 'a+')
  f.write(now_time+','+str(rgb_mean[0])+','+str(rgb_mean[1])+','+str(rgb_mean[2])+'\n')
  f.close()


  result=np.array([])
  qvec = np.array([rgb_mean[0],rgb_mean[1],rgb_mean[2]])
  for i in range(rgbs.index.shape[0]):
    result=np.append(result, comp_sim(qvec, rgbs.iloc[i,:].to_numpy()))
  rank=np.argsort(result)
  return_result = []
  return_filepath = []
  for index in rank[:-rank.shape[0]-1:-1]:
    return_result.append(result[index])
    return_filepath.append(tfilepaths[index])

  files = glob.glob("./static/tmp/*_akaze.jpg")

  if os.path.isfile("./static/output/metadata.pickle"):
    with open("./static/output/metadata.pickle", mode="rb") as f:
      sources = pickle.load(f)
  else:
    sources = {}
  akaze = cv2.AKAZE_create()
  kp, des = akaze.detectAndCompute(img, None)
  features = [kp, des]
  keypoints=[]
  kp_akaze = cv2.drawKeypoints(img, kp, None, flags=4)
  cv2.imwrite(f'./static/tmp/{now_time}_akaze.jpg', kp_akaze)
  for keys in features[0]:
    temp = (keys.pt, keys.size, keys.angle, keys.response, keys.octave, keys.class_id)
    keypoints.append(temp)
  # keypointsをbytesに変換
  map(bytes, keypoints)
  sources[f'./static/tmp/{now_time}.jpg'] = {
      "src": f'./static/tmp/{now_time}.jpg',
      "keypoint": keypoints,
      "descriptor": features[1],
      }

  with open("./static/output/metadata.pickle", mode="wb") as f:
    pickle.dump(sources, f)


  files = glob.glob("./static/tmp/*_akaze.jpg")
  json_data = json.dumps({"data":files,"rgbfile":return_filepath,"rgb_rank":return_result})
  return json_data

@app.route("/get_akaze",methods = ["POST"])
def get_akaze():
  files = glob.glob("./static/tmp/*_akaze.jpg")
  json_data = json.dumps({"data":files})
  return json_data

def comp_sim(qvec,tvec):
  return np.dot(qvec, tvec) / (np.linalg.norm(qvec) * np.linalg.norm(tvec))
## おまじない
if __name__ == "__main__":
  app.run(debug=True)