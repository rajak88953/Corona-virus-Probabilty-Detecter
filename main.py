from flask import Flask, render_template,request

#app = Flask(__name__, template_folder='../templates', static_folder='../static')
app = Flask(__name__)
import pickle

file=open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['fever'])
        pain = int(myDict['pain'])
        runnnyNose = int(myDict['runnnyNose'])
        diffBreath = int(myDict['diffBreath'])
        #code for infrence
        inputFeatures = [fever,pain,age,runnnyNose,diffBreath]
        infprob = clf.predict_proba([inputFeatures])[0][1]
        print(infprob)
        return render_template('show.html', inf=round(infprob*100))
    return render_template('index.html')
        #return 'Hello world!' + str(infprob)


if __name__=="__main__":
    app.run(debug=True)

