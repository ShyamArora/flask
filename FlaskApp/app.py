from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/showchat')
def showchat():
    return render_template('chat.html')

@app.route('/output')
def output():
    return render_template('output.html')

@app.route('/signUp',methods=['POST'])
def signUp():
    
    comment = request.form['inputName']
    print(comment)
    if comment:
        
        return json.dumps({'html':'<span>All fields good !!</span>'})
    else:
        return json.dumps({'html':'<span>Enter the required fields</span>'})
    return render_template('chat.html')



if __name__ == "__main__":
    app.run()


