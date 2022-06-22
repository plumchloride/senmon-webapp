from flask import Flask, render_template ,request

app = Flask(__name__)

url_address = ["0615_1","0615_2"]

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


## おまじない
if __name__ == "__main__":
  app.run(debug=True)