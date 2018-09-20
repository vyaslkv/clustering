from flask import Flask, jsonify, render_template, request
import MySQLdb, json
import requests
app = Flask(__name__)
# @app.route('/',methods=['GET', 'POST'])
# def index():
#     return render_template('form.html')

@app.route('/',methods=['GET', 'POST'])
def display():
    db = MySQLdb.connect("localhost","root","test123","test" )
    with db:
        cursor = db.cursor()
            
        sql = "SELECT * FROM geninfo1"
            
        cursor.execute(sql)
        rows = cursor.fetchall()
        x=[]
        for i, item in enumerate(rows):
            x.append({
                    'id':rows[i][0],
                    'Name':rows[i][1],
                    'Class':rows[i][3],
                    'Town':rows[i][4],
                    'Roll':rows[i][2],
                    'Remarks':rows[i][5],
                    })
    return jsonify(x)
            

      
    db.close()
@app.route('/update',methods=['GET', 'POST'])
def update():
    data=request.data
    dataDict = json.loads(data)
    db = MySQLdb.connect("localhost","root","test123","test" )
    with db:
        cursor = db.cursor()

        sql = "UPDATE geninfo1 SET Name=%s, Class=%s, Town=%s, Roll=%s, Remarks=%s WHERE id=%s"

        cursor.execute (sql, (dataDict['name1'], dataDict['class1'], dataDict['town1'], dataDict['roll1'], dataDict['remarks1'], dataDict['id1']))

    db.close()
@app.route('/add',methods=['GET', 'POST'])
def add():      
    data=request.data
    dataDict = json.loads(data)
    db = MySQLdb.connect("localhost","root","test123","test" )
    with db:
        cursor = db.cursor()

        sql = "INSERT INTO geninfo1 (Name, Roll, Class, Town, Remarks) VALUES (%s,%s,%s,%s,%s)"

        cursor.execute (sql, (dataDict['name1'], dataDict['roll1'], dataDict['class1'], dataDict['town1'], dataDict['remarks1']))

    db.close()
@app.route('/where',methods=['GET', 'POST'])
def where():
    data=request.data
    dataDict = json.loads(data)
    db = MySQLdb.connect("localhost","root","test123","test" )
    with db:
        cursor = db.cursor()

        sql = "SELECT * FROM geninfo1 WHERE Name=%s"

        cursor.execute (sql, (dataDict['name1'],))
        rows = cursor.fetchall()
        x=[] 
        for i, item in enumerate(rows):
            x.append({
                 'id' :rows[i][0],
                 'Name':rows[i][1],
                 'Roll':rows[i][3],
                 'Town':rows[i][4],
                 'Class':rows[i][2],
                 'Remarks':rows[i][5],
                    })
    return jsonify(x)            
        
    db.close()



if __name__=='__main__':
           app.run(host="10.2.10.55", port=5000, debug=True)
