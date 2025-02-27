from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'replace_with_a_secret_key'

# Connect to MongoDB using the URI from the environment variable
mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/SurveyApp')
client = MongoClient(mongo_uri)
db = client.get_default_database()

@app.route('/survey')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Gather form data
    data = {
        'name': request.form.get('name'),
        'email': request.form.get('email'),
        'image_included': request.form.getlist('image-included'),
        'other_details': request.form.get('other-details'),
        'pattern_percentage': request.form.get('pattern-percentage'),
        'pattern_found': request.form.get('pattern-found'),
        'search_rank': request.form.get('search-rank'),
        'tcc_pattern_number': request.form.get('tcc-pattern-number'),
        'general_feedback': request.form.get('general-feedback'),
        'submitted_at': datetime.utcnow()  # timestamp
    }

    # Insert the document into the surveys collection and get its _id
    result = db.surveys.insert_one(data)
    survey_id = str(result.inserted_id)

    flash("Survey submitted successfully!", "success")
    # Redirect to the results page with the survey_id as a query parameter
    return redirect(url_for('results', survey_id=survey_id))

@app.route('/results')
def results():
    survey_id = request.args.get('survey_id')
    if survey_id:
        # Retrieve only the survey with the given _id
        survey = db.surveys.find_one({"_id": ObjectId(survey_id)})
        return render_template('results.html', survey=survey)
    else:
        flash("No survey submission found.", "error")
        return redirect(url_for('index'))