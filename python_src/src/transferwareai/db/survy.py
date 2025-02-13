from pymongo import MongoClient
from datetime import datetime
from marshmallow import Schema, fields, validate, ValidationError

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.surveyApp
survey_collection = db.surveys

# Define the survey schema using Marshmallow
class SurveySchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1))
    email = fields.Email(required=True)
    searched_image_details = fields.Str(
        required=True,
        validate=validate.OneOf(["Border", "Maker's Mark", "Center Design", "Other"])
    )
    other_description = fields.Str(required=False, allow_none=True)
    pattern_found = fields.Str(required=True)
    pattern_percent = fields.Str(required=True)
    search_rank = fields.Str(required=False)
    general_feedback = fields.Str(required=True, validate=validate.Length(min=5))
    submitted_at = fields.DateTime(dump_only=True)

    # Custom validation: Ensure `other_description` is present if `sherd_portion` is "Other"
    def validate_other_description(self, data):
        if data["searched_image_details"] == "Other" and not data.get("other_description"):
            raise ValidationError("Other description is required when sherd portion is 'Other'.")
        return data

# Create an instance of the schema
survey_schema = SurveySchema()

# Example survey data
survey_data = {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "searched_image_details": "Border",
    "pattern_found": "1234",
    "pattern_percent": "TCC123",
    "general_feedback": "This app is fantastic!",
}

# Validate and insert the data
try:
    # Validate the data
    valid_data = survey_schema.load(survey_data)

    # Add submission timestamp
    valid_data["submitted_at"] = datetime.utcnow()

    # Insert into MongoDB
    result = survey_collection.insert_one(valid_data)
    print(f"Survey inserted with ID: {result.inserted_id}")
except ValidationError as err:
    print("Validation Error:", err.messages)
