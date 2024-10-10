import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class SpendingAnalyzer:
    def __init__(self):
        # Load CSV data
        self.load_data()
        self.prepare_data()
        self.train_model()

    def load_data(self):
        # Load CSV files
        self.account_df = pd.read_csv('data/account.csv')
        self.category_df = pd.read_csv('data/category.csv')
        self.date_df = pd.read_csv('data/date.csv')
        self.transaction_type_df = pd.read_csv('data/transaction_type.csv')
        self.transactions_df = pd.read_csv('data/transactions.csv')

    def prepare_data(self):
        # Merge dataframes
        merged_df = self.transactions_df.merge(self.date_df, on='date_id', how='left') \
            .merge(self.account_df, on='account_id', how='left') \
            .merge(self.category_df, on='category_id', how='left') \
            .merge(self.transaction_type_df, on='transaction_type_id', how='left')

        # Feature engineering
        merged_df['is_expense'] = merged_df['type'].apply(lambda x: 1 if x == 'expense' else 0)
        merged_df['amount'] = merged_df['amount'].abs()
        merged_df['month'] = merged_df['month'].astype(int)
        merged_df['day_of_week'] = merged_df['day_of_week'].astype(int)

        # Group by month and category for aggregated features
        monthly_expenditure = merged_df.groupby(['month', 'category_name'])['amount'].sum().reset_index()
        monthly_expenditure.rename(columns={'amount': 'total_monthly_expenditure'}, inplace=True)

        # Merge this back to the main dataframe
        self.merged_df = merged_df.merge(monthly_expenditure, on=['month', 'category_name'], how='left')

    def train_model(self):
        # Define features and target variable
        features = ['month', 'day_of_week', 'is_expense', 'total_monthly_expenditure']
        target = 'amount'

        X = self.merged_df[features]
        y = self.merged_df[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        self.model.fit(X_train, y_train)

    def analyze_spending(self, transactions):
        # Prepare input data
        input_df = pd.DataFrame(transactions)

        # Feature engineering for input data
        input_df['is_expense'] = input_df['transaction_type'].apply(lambda x: 1 if x == 'expense' else 0)
        input_df['month'] = pd.to_datetime(input_df['date']).dt.month
        input_df['day_of_week'] = pd.to_datetime(input_df['date']).dt.dayofweek

        # Use average monthly expenditure by category if necessary
        input_df = input_df.merge(self.merged_df[['category_name', 'total_monthly_expenditure']],
                                  on='category_name', how='left')

        # Make predictions
        predictions = self.model.predict(input_df[['month', 'day_of_week', 'is_expense', 'total_monthly_expenditure']])
        input_df['predicted_amount'] = predictions

        # Generate recommendations
        recommendations = self.generate_recommendations(input_df)

        # Prepare response
        response = input_df[['date', 'category_name', 'amount', 'predicted_amount']].to_dict(orient='records')
        return {
            "spending_analysis": response,
            "recommendations": recommendations
        }

    def generate_recommendations(self, input_df):
        recommendations = []

        # Calculate total spending
        total_spending = input_df['amount'].sum()
        total_predicted = input_df['predicted_amount'].sum()

        # Calculate the average spending by category
        category_spending = input_df.groupby('category_name').sum(numeric_only=True)['amount']
        category_predicted = input_df.groupby('category_name').sum(numeric_only=True)['predicted_amount']

        for category in category_spending.index:
            actual = category_spending[category]
            predicted = category_predicted[category]
            budgeted = actual / len(input_df) * 1.2  # Example budget threshold

            if actual > budgeted:
                recommendations.append(
                    f"You are overspending on {category}. Consider reducing your expenses in this category.")

        # General recommendation based on total spending
        if total_spending > total_predicted:
            recommendations.append("You're likely to overspend this month if you continue this spending pattern.")

        return recommendations
