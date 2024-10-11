from flask import Flask, render_template, jsonify, redirect, request
from models.model import SpendingAnalyzer
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

spending_analyzer = SpendingAnalyzer()

@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route('/api/analyze-spending', methods=['POST'])
def analyze_spending():
    # Get the JSON data from the request
    data = request.get_json()

    # Check if 'transactions' key is in the JSON data
    if 'transactions' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Get transactions from the input data
    transactions = data['transactions']

    # Analyze spending and get recommendations
    results = spending_analyzer.analyze_spending(transactions)

    # Return the results as JSON
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
