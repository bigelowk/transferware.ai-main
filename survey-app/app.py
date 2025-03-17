from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from datetime import datetime
import requests
import logging

app = Flask(__name__)
app.secret_key = 'replace_with_a_secret_key'

# Connect to MongoDB using the URI from the environment variable
mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/SurveyApp')
EXTERNAL_API_URL = "http://transferware-ai.umd.umich.edu/api/analytics_id/"
client = MongoClient(mongo_uri)
db = client.get_default_database()

@app.route('/')
def index():
    analytics_id = request.args.get('analytics_id') #get_analytics_id()
    return render_template('index.html')

@app.route('/survey/', methods=['GET'])
def survey():
    analytics_id = request.args.get('analytics_id')  # Get analytics_id from query parameters
    # Render the survey page, passing the analytics_id to the template
    return render_template('index.html', analytics_id=analytics_id)

@app.route('/submit', methods=['POST'])
def submit():
    # Gather form data
    #analytics_id = request.args.get('analytics_id') #get_analytics_id()
    #logging.info("REQID: {request_id}")
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
        'submitted_at': datetime.utcnow(),  # timestamp
        'analytics_id': request.form.get('analytics_id')

    }

    # Insert the document into the surveys collection and get its _id
    result = db.surveys.insert_one(data)
    survey_id = str(result.inserted_id)

    # Log the form data for debugging
    logging.info("Form Data: %s", data)

    try:
        result = db.surveys.insert_one(data)
        survey_id = str(result.inserted_id)
    except Exception as e:
        logging.error("Error inserting survey data: %s", e)
        flash("Error submitting the survey. Please try again.", "error")
        return redirect(url_for('index'))

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


def get_analytics_id():

    # Send a GET request to the external API
    response = requests.get(EXTERNAL_API_URL, timeout=10)

    # Check if the response was successful (status code 200)
    if response.status_code == 200:
        external_data = response # Convert response to JSON
        logging.info("REQID: {external_data}")
        result_id = external_data.get("result_id")
        logging.info("REQID: {result_id}")
        return result_id
            
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
