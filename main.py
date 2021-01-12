from flask import Flask,render_template,request
from joblib import load

clf = load('corona')
# print(clf)
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def coronapred():
    if request.method=='POST':
        mydict = request.form
        fever = int(mydict['fever'])
        cold = int(mydict['cold'])
        tiredness = int(mydict['tiredness'])
        sorethroat = int(mydict['sorethroat'])
        headache = int(mydict['headache'])
        loss_of_taste = int(mydict['loss_of_taste'])
        age = int(mydict['age'])
        features = [[fever,cold,tiredness,sorethroat,headache,loss_of_taste,age]]
        print(features)
        infprob = (clf.predict_proba(features))
        return render_template('show.html',inf=infprob[0][1])
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)



