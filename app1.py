from flask import Flask, render_template, request
import csv
# import cgi
# from model import SessionOutput
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import SearchModel
import pandas as pd
import json
# https://stackoverflow.com/questions/43677564/passing-input-from-html-to-python-and-back
app = Flask(__name__)

# load songs from CSV file
data_master = pd.read_json('lyrics_200.jl',lines= True)
data = data_master.copy()
print("Data read successfully")
print("Data read successfully")

# form = cgi.FieldStorage()
# searchterm =  form.getvalue('searchbox')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_input']
    model = SearchModel(data)
    result = model.matching(search_query)
    print("Result computed successfully")
    output = json.loads(result.to_json(orient="records"))
    return render_template('index1.html',output=output)


if __name__ == '__main__':
    app.run(debug=True)
