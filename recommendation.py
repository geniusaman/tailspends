import pandas as pd
import streamlit as st
import textwrap
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import trim_mean
import pandas as pd
import textwrap
import json
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize the Groq client (your API key should remain hidden in production)
# client = Groq(
#     api_key='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',
# )
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# Initialize LLM
groq_llm = ChatGroq(
model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY,
)

# co = cohere.Client('n67TO7BAR9zm3ZZe8duwE7PGf8LVQilR7oNBZA9R')
 

def calculate_trimmed_mean(x):
    """
    Calculate trimmed mean (5% trimming)
    """
    return trim_mean(x, proportiontocut=0.10)

def process_supplier_data(df1, df2):
    # Concatenate the two DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(combined_df.columns)

    # Calculate the trimmed mean of 'Average Unit Price' for the entire dataset
    market_price = calculate_trimmed_mean(combined_df['Average Unit Price($)'])

    # Add the calculated 'Market Price' as a new column to the combined DataFrame
    combined_df['Market Price'] = market_price

    # Calculate Proposed Price Lower and Upper Bounds
    combined_df['Proposed Price Lower Bound'] = combined_df['Market Price'] * 0.90
    combined_df['Proposed Price Upper Bound'] = combined_df['Market Price'] * 1.10

    # Calculate projected savings for lower bound Proposed Price Upper Bound
    combined_df['Projected Savings Lower'] = (
        (combined_df['Average Unit Price($)'] * combined_df['Total Quantity']) -
        (combined_df['Proposed Price Lower Bound'] * combined_df['Total Quantity'])
    )

    # Calculate projected savings for upper bound
    combined_df['Projected Savings Upper'] = (
        (combined_df['Average Unit Price($)'] * combined_df['Total Quantity']) -
        (combined_df['Proposed Price Upper Bound'] * combined_df['Total Quantity'])
    )

    return combined_df
def analyze_and_display_suppliers(df, threshold=0.15):
    st.subheader("Choose the weightage")

    col1, col2, col3 = st.columns(3)

    # Place the first three sliders in the first row
    with col1:
        rankings = {
            'Average Unit Price($)': st.slider("Average Unit Price", min_value=1, max_value=5, value=4, key='price_slider')
        }
    with col2:
        rankings.update({
            'Total Spend($)': st.slider("Total Spend", min_value=1, max_value=5, value=5, key='spend_slider')
        })
    with col3:
        rankings.update({
            'Total Quantity': st.slider("Total Quantity", min_value=1, max_value=5, value=2, key='quantity_slider')
        })

    # Create two columns for the second row
    col4, col5 = st.columns(2)

    # Place the remaining sliders in the second row
    with col4:
        rankings.update({
            'Transactions Count': st.slider("Transactions Count", min_value=1, max_value=5, value=1, key='transactions_slider')
        })
    with col5:
        rankings.update({
            'Supplier Performance Score': st.slider("Supplier Performance Score", min_value=1, max_value=5, value=3, key='performance_slider')
        })

    # Columns to normalize
    columns = ['Average Unit Price($)', 'Total Spend($)', 'Total Quantity',
               'Transactions Count', 'Supplier Performance Score']

    # Normalize the data using Min-Max scaling (0 to 1)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns])

    # Create new columns for normalized values
    for i, col in enumerate(columns):
        df[f'Normalized {col}'] = normalized_data[:, i]

    # Invert the "Average Unit Price" since lower prices are better
    df['Normalized Average Unit Price($)'] = 1 - df['Normalized Average Unit Price($)']

    # Convert rankings to weights (rank 1 has the highest weight, and rank 5 has the lowest)
    max_rank = max(rankings.values())
    weights = {col: (max_rank + 1 - rank) for col, rank in rankings.items()}

    # Normalize the weights so that their sum equals 1
    total_weight = sum(weights.values())
    normalized_weights = {col: weight / total_weight for col, weight in weights.items()}

    # Calculate the weighted score using normalized data and normalized weights
    df['Weighted Score'] = sum(df[f'Normalized {col}'] * normalized_weights[col] for col in columns)

    # Filter suppliers that meet the threshold condition
    consolidated_suppliers = df[df['Weighted Score'] > threshold]
    eliminated_suppliers = df[df['Weighted Score'] <= threshold]

    # Calculate percentiles dynamically
    percentiles = {
        'Average Unit Price($)': df['Average Unit Price($)'].rank(pct=True),
        'Total Spend($)': df['Total Spend($)'].rank(pct=True),
        'Total Quantity': df['Total Quantity'].rank(pct=True),
        'Transactions Count': df['Transactions Count'].rank(pct=True),
        'Supplier Performance Score': df['Supplier Performance Score'].rank(pct=True)
    }

    # Identify reasons for consolidation and elimination dynamically
    consolidated_suppliers_data = []
    eliminated_suppliers_data = []

    # Loop through consolidated suppliers and dynamically determine their strengths
    for _, supplier in consolidated_suppliers.iterrows():
        reasons = []
        if percentiles['Average Unit Price($)'][supplier.name] <= 0.40:
            reasons.append(f"Competitive Average Unit Price (\${supplier['Average Unit Price($)']:,.2f})")
        if percentiles['Total Spend($)'][supplier.name] >= 0.40:
            reasons.append(f"High Total Spend (\${supplier['Total Spend($)']:,.2f})")
        if percentiles['Total Quantity'][supplier.name] >= 0.40:
            reasons.append(f"High Total Quantity ({int(supplier['Total Quantity'])})")
        if percentiles['Transactions Count'][supplier.name] >= 0.40:
            reasons.append(f"Higher Transactions Count ({int(supplier['Transactions Count'])})")
        if percentiles['Supplier Performance Score'][supplier.name] >= 0.40:
            reasons.append(f"Good Supplier Performance Score ({supplier['Supplier Performance Score']:.6f})")
        if reasons:
            consolidated_suppliers_data.append([supplier['Supplier Name'], reasons])

    # Loop through eliminated suppliers and dynamically determine their weaknesses
    for _, supplier in eliminated_suppliers.iterrows():
        reasons = []
        if percentiles['Average Unit Price($)'][supplier.name] >= 0.50:
            reasons.append(f"High Average Unit Price (\${supplier['Average Unit Price($)']:,.2f})")
        if percentiles['Total Spend($)'][supplier.name] <= 0.50:
            reasons.append(f"Low Total Spend (\${supplier['Total Spend($)']:,.2f})")
        if percentiles['Total Quantity'][supplier.name] <= 0.50:
            reasons.append(f"Low Total Quantity ({int(supplier['Total Quantity'])})")
        if percentiles['Transactions Count'][supplier.name] <= 0.50:
            reasons.append(f"Lower Transactions Count ({int(supplier['Transactions Count'])})")
        if percentiles['Supplier Performance Score'][supplier.name] <= 0.50:
            reasons.append(f"Poor Supplier Performance Score ({supplier['Supplier Performance Score']:.6f})")
        if reasons:
            eliminated_suppliers_data.append([supplier['Supplier Name'], reasons])

    def format_supplier_score(score):
        return f"{round(score)}/10"

    # Commented out financial score formatting
    # def format_financial_score(score):
    #     return f"{round(score)}/5"

    def create_markdown_table(data, headers):
        markdown = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
        for row in data:
            markdown += f"| {row[0]} | {', '.join(row[1])} |\n"
        return markdown

    # Display Consolidated Suppliers table
    st.subheader("1. Consolidated suppliers and reasons")
    consolidated_table = create_markdown_table(consolidated_suppliers_data, ["Supplier Name", "Reasons"])
    cons_sup = consolidated_suppliers.reset_index(drop=True)
    cons_sup['Supplier Performance Score'] = cons_sup['Supplier Performance Score'].apply(format_supplier_score)
    # Removed financial score formatting
    # cons_sup['Financial Score'] = cons_sup['Financial Score'].apply(format_financial_score)

    # Display only the desired columns for consolidated suppliers (removed Financial Score)
    st.dataframe(cons_sup[['Supplier Name', 'Category', 'Subcategory', 'Product', 'Average Unit Price($)', 
                          'Total Spend($)', 'Total Quantity', 'Transactions Count', 
                          'Supplier Performance Score', 'Contract Status', 
                          'Contract ID', 'Contract Total Value', 'Payment Terms']], hide_index=True)
    st.markdown(consolidated_table)

    # # Display Eliminated Suppliers table
    # st.subheader("2. Eliminated suppliers and reasons")
    # eliminated_table = create_markdown_table(eliminated_suppliers_data, ["Supplier Name", "Reasons"])
    # elm_sup = eliminated_suppliers.reset_index(drop=True)
    # elm_sup['Supplier Performance Score'] = elm_sup['Supplier Performance Score'].apply(format_supplier_score)
    # elm_sup['Financial Score'] = elm_sup['Financial Score'].apply(format_financial_score)

    # # Display only the desired columns for eliminated suppliers
    # st.dataframe(elm_sup[['Supplier Name', 'Category', 'Subcategory', 'Product', 'Average Unit Price($)', 
    #                      'Total Spend($)', 'Total Quantity', 'Transactions Count', 
    #                      'Supplier Performance Score', 'Financial Score', 'Contract Status', 
    #                      'Contract ID', 'Contract Total Value', 'Payment Terms']], hide_index=True)
    # # st.markdown(eliminated_table)
    # st.markdown("""
    # **Retained Suppliers:**
    # *  Turner and Allen & Roberts-Ferguson: These suppliers are recommended for retention due to their strong performance scores, and favorable payment terms with early payment discounts. Turner and Allen provides the best discount of 4.7%, while Roberts-Ferguson demonstrates reliability with multiple transactions.

    # **Eliminated Suppliers:**
    # * Lewis-Williams & Martin-White: These suppliers are recommended for elimination due to their higher unit prices (\\$1,290 and \\$1,300 respectively), less attractive early payment discounts, and lower transaction volumes compared to the retained suppliers.
    # """)
        
    return consolidated_suppliers, eliminated_suppliers

# def analyze_and_display_suppliers(df, threshold=0.15):
#     st.subheader("Choose the weightage")



#     col1, col2, col3 = st.columns(3)

#     # Place the first three sliders in the first row
#     with col1:
#         rankings = {
#             'Average Unit Price': st.slider("Average Unit Price", min_value=1, max_value=5, value=4, key='price_slider')
#         }
#     with col2:
#         rankings.update({
#             'Total Spend': st.slider("Total Spend", min_value=1, max_value=5, value=5, key='spend_slider')
#         })
#     with col3:
#         rankings.update({
#             'Total Quantity': st.slider("Total Quantity", min_value=1, max_value=5, value=2, key='quantity_slider')
#         })

#     # Create two columns for the second row
#     col4, col5 = st.columns(2)

#     # Place the remaining sliders in the second row
#     with col4:
#         rankings.update({
#             'Transactions Count': st.slider("Transactions Count", min_value=1, max_value=5, value=1, key='transactions_slider')
#         })
#     with col5:
#         rankings.update({
#             'Supplier Performance Score': st.slider("Supplier Performance Score", min_value=1, max_value=5, value=3, key='performance_slider')
#         })

#         # Close the custom container div
#     # Columns to normalize
#     columns = ['Average Unit Price', 'Total Spend', 'Total Quantity',
#                'Transactions Count', 'Supplier Performance Score']

#     # Normalize the data using Min-Max scaling (0 to 1)
#     scaler = MinMaxScaler()
#     normalized_data = scaler.fit_transform(df[columns])

#     # Create new columns for normalized values
#     for i, col in enumerate(columns):
#         df[f'Normalized {col}'] = normalized_data[:, i]

#     # Invert the "Average Unit Price" since lower prices are better
#     df['Normalized Average Unit Price'] = 1 - df['Normalized Average Unit Price']
    
#     # Convert rankings to weights (rank 1 has the highest weight, and rank 5 has the lowest)
#     max_rank = max(rankings.values())
#     weights = {col: (max_rank + 1 - rank) for col, rank in rankings.items()}

#     # Normalize the weights so that their sum equals 1
#     total_weight = sum(weights.values())
#     normalized_weights = {col: weight / total_weight for col, weight in weights.items()}

#     # Calculate the weighted score using normalized data and normalized weights
#     df['Weighted Score'] = sum(df[f'Normalized {col}'] * normalized_weights[col] for col in columns)

#     # Filter suppliers that meet the threshold condition
#     consolidated_suppliers = df[df['Weighted Score'] > threshold]
#     eliminated_suppliers = df[df['Weighted Score'] <= threshold]

#     # Identify reasons for consolidation and elimination dynamically
#     consolidated_suppliers_data = []
#     eliminated_suppliers_data = []

#     # Loop through consolidated suppliers and dynamically determine their strengths
#     for _, supplier in consolidated_suppliers.iterrows():
#         reasons = []
#         if supplier['Average Unit Price'] < df['Average Unit Price'].mean():
#             reasons.append(f"Competitive Average Unit Price (\${supplier['Average Unit Price']:,.2f})")
#         if supplier['Total Spend'] >= df['Total Spend'].mean():
#             reasons.append(f"High Total Spend (\${supplier['Total Spend']:,.2f})")
#         if supplier['Total Quantity'] >= df['Total Quantity'].mean():
#             reasons.append(f"High Total Quantity ({int(supplier['Total Quantity'])})")
#         if supplier['Transactions Count'] >= df['Transactions Count'].mean():
#             reasons.append(f"Higher Transactions Count ({int(supplier['Transactions Count'])})")
#         if supplier['Supplier Performance Score'] >= df['Supplier Performance Score'].mean():
#             reasons.append(f"Good Supplier Performance Score ({supplier['Supplier Performance Score']:.6f})")
#         consolidated_suppliers_data.append([supplier['Supplier Name'], reasons])

#     # Loop through eliminated suppliers and dynamically determine their weaknesses
#     for _, supplier in eliminated_suppliers.iterrows():
#         reasons = []
#         if supplier['Average Unit Price'] > df['Average Unit Price'].mean():
#             reasons.append(f"High Average Unit Price (\${supplier['Average Unit Price']:,.2f})")
#         if supplier['Total Spend'] < df['Total Spend'].mean():
#             reasons.append(f"Low Total Spend (\${supplier['Total Spend']:,.2f})")
#         if supplier['Total Quantity'] < df['Total Quantity'].mean():
#             reasons.append(f"Low Total Quantity ({int(supplier['Total Quantity'])})")
#         if supplier['Transactions Count'] < df['Transactions Count'].mean():
#             reasons.append(f"Lower Transactions Count ({int(supplier['Transactions Count'])})")
#         if supplier['Supplier Performance Score'] < df['Supplier Performance Score'].mean():
#             reasons.append(f"Poor Supplier Performance Score ({supplier['Supplier Performance Score']:.6f})")
#         eliminated_suppliers_data.append([supplier['Supplier Name'], reasons])

#     # # Display Consolidated Suppliers and Reasons
#     # st.subheader("1. Consolidated suppliers and reasons")
#     # for supplier in consolidated_suppliers_data:
#     #     st.write(f"**Supplier Name**: {supplier[0]}")
#     #     st.write(f"**Reasons**: {', '.join(supplier[1])}")

#     # # Display Eliminated Suppliers and Reasons
#     # st.subheader("2. Eliminated suppliers and reasons")
#     # for supplier in eliminated_suppliers_data:
#     #     st.write(f"**Supplier Name**: {supplier[0]}")
#     #     st.write(f"**Reasons**: {', '.join(supplier[1])}")
#     # Convert consolidated suppliers data to a DataFrame
    def format_supplier_score(score):
        return f"{round(score)}/10"

    # Function to format Financial Score
    def format_financial_score(score):
        return f"{round(score)}/5"

    def create_markdown_table(data, headers):
        markdown = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
        for row in data:
            markdown += f"| {row[0]} | {', '.join(row[1])} |\n"
        return markdown
    

    # Display Consolidated Suppliers table
    st.subheader("1. Consolidated suppliers and reasons")
    consolidated_table = create_markdown_table(consolidated_suppliers_data, ["Supplier Name", "Reasons"])
    cons_sup = consolidated_suppliers.reset_index(drop=True)
    cons_sup['Supplier Performance Score'] = cons_sup['Supplier Performance Score'].apply(format_supplier_score)
    cons_sup['Financial Score'] = cons_sup['Financial Score'].apply(format_financial_score)
    cons_sup = cons_sup.rename(columns={
        'Average Unit Price': 'Average Unit Price($)',
        'Total Spend': 'Total Spend($)',
    })
    st.dataframe(cons_sup,hide_index=True,)
    st.markdown(consolidated_table)

    # Display Eliminated Suppliers table
    st.subheader("2. Eliminated suppliers and reasons")
    eliminated_table = create_markdown_table(eliminated_suppliers_data, ["Supplier Name", "Reasons"])
    elm_sup = eliminated_suppliers.reset_index(drop=True)
    elm_sup['Supplier Performance Score'] = elm_sup['Supplier Performance Score'].apply(format_supplier_score)
    elm_sup['Financial Score'] = elm_sup['Financial Score'].apply(format_financial_score)
    elm_sup = elm_sup.rename(columns={
        'Average Unit Price': 'Average Unit Price($)',
        'Total Spend': 'Total Spend($)',
    })
    st.dataframe(elm_sup,hide_index=True,)
    st.markdown(eliminated_table)
        
    # st.write(consolidated_suppliers)
    
    # Return the DataFrames of consolidated and eliminated suppliers
    return consolidated_suppliers, eliminated_suppliers
    # return consolidated_suppliers_data, eliminated_suppliers_data['Total Spend'].sum()


def format_with_commas(value):
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return value

def display_negotiation_recommendations(df1):
    # Define the percentage range for negotiation (e.g., 5-15% lower)
    negotiation_upper_pct = 0.15  # 15% lower

    # Recalculate the negotiated price based on the Average Unit Price
    df1['Negotiated Price (Lower Bound)'] = df1['Average Unit Price($)'] * (1 - negotiation_upper_pct)
    df1['Negotiated Price (Upper Bound)'] = df1['Market Price']
    
    filtered_df = df1.copy()
    # Filtering suppliers where 'Average Unit Price' > 'Market Price'
    filtered_df = df1[df1['Average Unit Price($)'] > df1['Market Price']]

    # Calculate total cost at market price
    total_cost_of_market_price = filtered_df['Market Price'] * (filtered_df['Total Spend($)'] / filtered_df['Average Unit Price($)'])
    
    # Calculate Profit Saved Total as Total Spend - total_cost_of_market_price
    filtered_df['Profit Saved Total'] = filtered_df['Total Spend($)'] - total_cost_of_market_price
    
    filtered_df['Profit Saved Total'] = format_with_commas(filtered_df['Profit Saved Total'].round(2))
    # Formatting 'Market Price' column to display whole numbers
    filtered_df['Market Price'] = format_with_commas(filtered_df['Market Price'].astype(int))

     # Calculate quantities and potential savings per supplier
    # filtered_df['Quantity'] = filtered_df['Total Spend($)'] / filtered_df['Average Unit Price($)']
    # filtered_df['Potential Savings'] = filtered_df['Quantity'] * (filtered_df['Average Unit Price($)'] - filtered_df['Market Price'])


    filtered_df['Recommendation'] = filtered_df.apply(lambda row: textwrap.dedent(f"""
    Supplier {row['Supplier Name']} is offering the product at \${row['Average Unit Price($)']:.2f} per 0.5 tone,
    which is higher than the market price of \$565 per 0.5 tone.
    We recommend negotiating with {row['Supplier Name']} to a price of \$565 per 0.5 tone, with the upper bound being equal to the Market Price. 
    This price reduction is recommended because it provides a fair negotiating range, considering market rates and room for a supplier discount.
    By doing so, we could save approximately \${format_with_commas(row['Profit Saved Total'])} on a total spend of \${format_with_commas(row['Total Spend($)'])}.
    """).strip(), axis=1)

    # Selecting the desired columns for output
    output = filtered_df[['Supplier Name', 'Average Unit Price($)', 'Market Price', 'Negotiated Price (Lower Bound)', 
                          'Negotiated Price (Upper Bound)', 'Total Spend($)', 'Profit Saved Total', 'Recommendation']]
    
    output = output.reset_index(drop=True)
    
    output1 = filtered_df[['Supplier Name', 'Average Unit Price($)', 'Market Price', 'Total Spend($)', 'Profit Saved Total']]
    output1 = output1.reset_index(drop=True)
    numeric_columns = ['Average Unit Price($)', 'Market Price', 
                    'Negotiated Price (Upper Bound)', 'Total Spend($)']

    output1_styled = output1.style.format({col: format_with_commas for col in numeric_columns})
    # Displaying the results in Streamlit
    st.subheader("2. Price negotiation recommendations")

    def df_to_markdown(df, format_dict=None):
        # Apply formatting if provided
        if format_dict:
            formatted_df = df.copy()
            for col, format_func in format_dict.items():
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(format_func)
        else:
            formatted_df = df

        # Create a list to store markdown rows
        markdown_rows = []
        
        # Add header
        header = "| " + " | ".join(formatted_df.columns) + " |"
        markdown_rows.append(header)
        
        # Add separator
        separator = "|" + "|".join(["---"] * len(formatted_df.columns)) + "|"
        markdown_rows.append(separator)
        
        # Add data rows
        for _, row in formatted_df.iterrows():
            markdown_row = "| " + " | ".join(row.astype(str)) + " |"
            markdown_rows.append(markdown_row)
        
        # Join all rows into a single string
        markdown_table = "\n".join(markdown_rows)
        return markdown_table

    # output1['Market Price'] = 565
    # total_cost_of_market_price = filtered_df['Market Price'] * (filtered_df['Total Spend($)'] / filtered_df['Average Unit Price($)'])
    # output1['Profit Saved Total'] = output1['Total Spend($)'] - total_cost_of_market_price
    # # Define the formatting for numeric columns
    # output1 = output1.rename(columns={'Profit Saved Total': 'Potential Cost Saving($)'})
    # output1 = output1.rename(columns={'Market Price': 'Market Price($)'})
    # numeric_columns = ['Average Unit Price($)', 'Market Price($)', 'Total Spend($)', 'Potential Cost Saving($)']
    # format_dict = {col: format_with_commas for col in numeric_columns}
    # Fix: Calculate the quantities for each supplier
    # output1['Quantity'] = output1['Total Spend($)'] / output1['Average Unit Price($)']
    
    # Set the fixed market price
    market_price = 565
    output1['Market Price'] = market_price
    
    # # Calculate potential cost savings using quantities
    # output1['Potential Cost Saving($)'] = output1['Quantity'] * (output1['Average Unit Price($)'] - market_price)
    
    # Format the numeric columns
    output1 = output1.rename(columns={'Profit Saved Total': 'Potential Cost Saving($)'})
    output1 = output1.rename(columns={'Market Price': 'Market Price($)'})
    numeric_columns = ['Average Unit Price($)', 'Market Price($)', 'Total Spend($)', 'Potential Cost Saving($)']
    format_dict = {col: format_with_commas for col in numeric_columns}
    # Convert the DataFrame to markdown
    markdown_table = df_to_markdown(output1, format_dict)

    st.markdown(markdown_table)
    # Loop through each row to show individual recommendations
    for _, row in output.iterrows():
        st.markdown(f"**Supplier Name**: {row['Supplier Name']}")
        st.markdown(f"**Recommendation**: {row['Recommendation']}")


# def negotiate_payment_terms(df: pd.DataFrame):
#     """
#     This function performs the following:
#     1. Extracts discount percentage and payment deadline from 'Payment Terms'.
#     2. Categorizes suppliers based on their financial score into 'Low', 'Medium', or 'High'.
#     3. Calculates a negotiated payment terms range based on financial score categories.
#     4. Generates a negotiation summary based on the supplier's financial score and proposed payment terms.
   
#     Args:
#     df (pd.DataFrame): DataFrame containing 'Payment Terms' and 'Financial Score' columns.
 
#     Returns:
#     None: Outputs the negotiation summary.
#     """
   
#     def extract_discount(text):
#         discount_pattern = r"(\d+(\.\d+)?)%"
#         discount_match = re.search(discount_pattern, text)
#         if discount_match:
#             return discount_match.group(1)
#         else:
#             return math.nan  # If no discount percentage is found
 
#     def extract_deadline(text):
#         deadline_pattern = r"(\d+)\sdays"
#         deadline_match = re.search(deadline_pattern, text)
#         if deadline_match:
#             return deadline_match.group(1)
#         else:
#             return math.nan  # If no payment deadline is found
 
#     # Extract discount and payment deadlines
#     df = df.assign(
#         Discount_Percent = df['Payment Terms'].map(extract_discount),
#         Payment_Deadline = df['Payment Terms'].map(extract_deadline)
#     )
 
#     # Convert 'Payment_Deadline' to float
#     df['Payment_Deadline'] = df['Payment_Deadline'].astype(str).str.extract('(\d+)').astype(float)
 
#     # Calculate percentiles for financial score
#     percentiles = df['Financial Score'].quantile([0.25, 0.50, 0.75])
#     p25, p50, p75 = percentiles[0.25], percentiles[0.50], percentiles[0.75]
 
#     # Categorize financial scores
#     def categorize_financial_score(score):
#         if score <= p25:
#             return 'Low'
#         elif p25 < score <= p75:
#             return 'Medium'
#         else:
#             return 'High'
 
#     df['Financial Score Category'] = df['Financial Score'].apply(categorize_financial_score)
 
#     def calculate_negotiated_payment_terms(row):
#         payment_terms = row['Payment_Deadline']
#         score_category = row['Financial Score Category']

#         if score_category == 'High':
#             lower_bound = payment_terms * 1.10
#             upper_bound = payment_terms * 1.20
#         elif score_category == 'Medium':
#             lower_bound = payment_terms * 1.05
#             upper_bound = payment_terms * 1.10
#         else:  # Low category
#             lower_bound = payment_terms
#             upper_bound = payment_terms

#         return f"{int(lower_bound)} - {int(upper_bound)} days"

#     # Ensure 'Payment_Deadline' and 'Financial Score Category' columns exist and are correctly formatted
#     if 'Payment_Deadline' in df.columns and 'Financial Score Category' in df.columns:
#         df['Negotiated Payment Terms Range'] = df.apply(calculate_negotiated_payment_terms, axis=1)

#         # Create the final DataFrame for negotiation summary
#         payment_term_negotiate = df[['Supplier Name', 'Payment Terms', 'Financial Score', 'Negotiated Payment Terms Range']]
#     else:
#         st.error("Error: Required columns are missing.")

 
#     # Create the final DataFrame for negotiation summary
#     payment_term_negotiate = df[['Supplier Name', 'Payment Terms', 'Financial Score', 'Negotiated Payment Terms Range']]
def negotiate_payment_terms(df: pd.DataFrame):
    """
    This function performs the following:
    1. Extracts discount percentage and payment deadline from 'Payment Terms'.
    2. Categorizes suppliers based on their financial score into 'Low', 'Medium', or 'High'.
    3. Calculates a negotiated payment terms range based on financial score categories.
    4. Calculates the potential cash flow impact of the new payment term using total spend as yearly spend.
    5. Generates a negotiation summary including the cash flow impact.
   
    Args:
    df (pd.DataFrame): DataFrame containing 'Supplier Name', 'Payment Terms', 'Financial Score', and 'Total Spend' columns.
 
    Returns:
    pd.DataFrame: Outputs the negotiation summary with cash flow impact.
    """
   
    import re
    import math

    def extract_discount(text):
        discount_pattern = r"(\d+(\.\d+)?)%"
        discount_match = re.search(discount_pattern, text)
        if discount_match:
            return float(discount_match.group(1))
        else:
            return math.nan  # If no discount percentage is found
 
    def extract_deadline(text):
        deadline_pattern = r"(\d+)\sdays"
        deadline_match = re.search(deadline_pattern, text)
        if deadline_match:
            return int(deadline_match.group(1))
        else:
            return math.nan  # If no payment deadline is found
 
    # Extract discount and payment deadlines
    df = df.assign(
        Discount_Percent = df['Payment Terms'].map(extract_discount),
        Payment_Deadline = df['Payment Terms'].map(extract_deadline)
    )
 
    # Convert 'Payment_Deadline' to float
    df['Payment_Deadline'] = df['Payment_Deadline'].astype(float)
 
    # Calculate percentiles for financial score
    percentiles = df['Financial Score'].quantile([0.25, 0.50, 0.75])
    p25, p50, p75 = percentiles[0.25], percentiles[0.50], percentiles[0.75]
 
    # Categorize financial scores
    def categorize_financial_score(score):
        if score <= p25:
            return 'Low'
        elif p25 < score <= p75:
            return 'Medium'
        else:
            return 'High'
 
    df['Financial Score Category'] = df['Financial Score'].apply(categorize_financial_score)
 
    def calculate_negotiated_payment_terms(row):
        payment_terms = row['Payment_Deadline']
        score_category = row['Financial Score Category']

        if score_category == 'High':
            lower_bound = payment_terms * 1.05
            upper_bound = payment_terms * 1.10
        elif score_category == 'Medium':
            lower_bound = payment_terms * 1.15
            upper_bound = payment_terms * 1.20
        else:  # Low category
            lower_bound = payment_terms * 1.30
            upper_bound = payment_terms * 1.35

        return int(lower_bound), int(upper_bound)
        
        # if score_category == 'High':
        #     return "High"
        # elif score_category == 'Medium':
        #     return "Medium"
        # else:  # Low category
        #     return "Low"

        # return score_category

    # Apply the payment terms negotiation
    df['Negotiated Payment Terms Range'] = df.apply(
        lambda row: calculate_negotiated_payment_terms(row), axis=1
    )
    
    # Extract lower bound of negotiated payment term for cash flow impact calculation
    df['New Payment Term'] = df['Negotiated Payment Terms Range'].apply(lambda x: x[0])
    # df['New Payment Term'] = df['Negotiated Payment Terms Range'].apply(lambda x: x[0])

    # Treat total spend as the average yearly spend (since you have data for one year)
    df['Average Yearly Spend'] = df['Total Spend($)']

    # Calculate the cash flow impact
    def calculate_cash_flow_impact(row):
        new_payment_term = row['New Payment Term']
        current_payment_term = row['Payment_Deadline']
        yearly_spend = row['Average Yearly Spend']
        
        # Formula: Cash Flow Impact = (New Payment Term - Current Payment Term) * Average Yearly Spend
        return (new_payment_term - current_payment_term) * yearly_spend

    df['Cash Flow Impact'] = df.apply(calculate_cash_flow_impact, axis=1)

    # Final negotiation summary
    payment_term_negotiate = df[['Supplier Name', 'Payment Terms', 'Financial Score', 'Negotiated Payment Terms Range', 'Cash Flow Impact']]
    
    # st.dataframe(payment_term_negotiate)

    df['Temp Range'] = df['Financial Score'].apply(categorize_financial_score)

    temp_payment_term_negotiate = df[['Supplier Name', 'Payment Terms', 'Financial Score', 'Temp Range', 'Cash Flow Impact', 'Total Spend($)']]

    # payment_term_negotiate_str = payment_term_negotiate.to_string(index=False)
    payment_term_negotiate_str = "\n".join([
        f"Supplier: {row['Supplier Name']}\n"
        f"Payment Terms: {row['Payment Terms']}\n"
        f"Financial Score: {row['Financial Score']}\n"
        f"Total Spend($): {row['Total Spend($)']}\n"
        f"Negotiation Range: {row['Temp Range']}\n"
        f"Cash Flow Impact: {format_with_commas(row['Cash Flow Impact'])}\n"
        f"{'-'*40}"  # Adds a separator between entries
        for idx, row in temp_payment_term_negotiate.iterrows()
    ])

    # st.write(payment_term_negotiate_str)

    # client = Groq(api_key='AIzaSyBiTCrxVaDb-Klv0lXLnF-ajpcankfoGdA')
 
    # Generate negotiation summary via Groq API
    # prompt = f"""
#  Supplier Payment Terms Negotiation Analysis
                
#     (Background - You are an AI assistant tasked with analyzing supplier payment terms and providing specific negotiation recommendations. Your goal is to suggest new payment terms for each supplier and calculate the potential cash flow impact based on the given data.)

#         Input data: {processed_data}(The input data is in tabular form)
#         Extract the following from input data (Take it as a table and carefully take all the columns correctly and do not change any values):

#         You need to provide a recommendation for the new payment term based on the current Payment term and Financial Score where the new payment term can only be increased by 15 or 30 days.
        
#         - Based on the supplier's Financial Score and Financial Score Category, recommend a specific new payment term based on the following rules:
        
#             - Ensure the recommended term is longer than the current Payment_Deadline.

#             - In case of Low Financial Score and Low Current Payment Term(for eg- 30 days) recommend 45 - 60 days.

#             - In case of Medium Financial Score and Medium Current Payment Term(for eg- 60 days) recommend 75 days.

#             - In case of High Financial Score and High Current Payment Term do not increase the payment term.

#             - The increase in Recommended payment term should strictly take place in +15 or +30 days(no other terms should be added such as 10,7,5 days) from the current Payment_Deadline.

#             - In case of Low Financial Score and High Current Payment Term do not increase the payment term.

#             - In case of Medium Financial Score and Low Current Payment Term increase the payment term by 15 days.

#             - In case of High Financial Score and Medium Current Payment Term do not increase the payment term.
#         ## Output Format must be:
        
#         Present your analysis as a concise, tabular. Structure your response as follows:(just provide the values in the table)
#         columns are,
#             1. Supplier Name 
            
#             2. Current Payment Term (days)
            
#             3. Recommended New Payment Term (days) (if current payment tern and recommended payment term is same then give the value as "Nil")
            
#             4. Potential Cash Flow Impact ($) (if current payment tern and recommended payment term is same then give the value as "0")
            
#             5. Financial Score (Represent financial score in decimal format with upto two digits precision)
        
#         After the tabular, provide a brief overall summary highlighting:
        
#         1. Total potential cash flow impact across all suppliers
        
#         2. Number of suppliers for which term extensions are recommended
        
#         3. Any suppliers where negotiation is not recommended (if applicable)
        
#         Provide a payment term negotiation in the end.  

#         - Remember to base all your analysis and recommendations strictly on the information provided in the DataFrame. If you encounter any ambiguities or need additional information to make a recommendation, state this clearly in your response.

#         - Provide bullet point for every recommendation.

#         ## Important Notes
        
#         - Do not invent or assume any data not present in the DataFrame.

#         - Carefully access all the Input Data before processing.
        
#         - If the Negotiated Payment Terms Range is not provided for a supplier, state that negotiation cannot be recommended due to lack of data.
        
#         - Round all financial figures to the nearest dollar.

#         - Do not display any rules or logic in your response from the prompt.
        
        ## Do not include code snippets"""

    example_json = '''
    { "suppliers": [ { "supplier_name": "", "total_spend($)": , "current_payment_term_days": , "negotiation_range": "", "potential_cash_flow_impact": , "payment_terms": "" } ] }

    '''
    prompt = (
         f"""
    Supplier Payment Terms Negotiation Analysis

    Input data: {payment_term_negotiate_str}
    Extract the content from the input data and return the response in JSON format.

    Extract the following from input data (Take it as a table and carefully take all the columns correctly and do not change any values):
    Feild Names:
        - Supplier Name(supplier_name as field name in response)
        - Total Spend ($)(total_spend as field name in response)
        - Current Payment Term (days)(just include the NET number of days integer value)(current_payment_term_days as field name in response)
        - Negotiotion Range (low, medium, high)(strictly keep "negotiation_range" as field name in response)
        - Potential Cash Flow Impact ($)(potential_cash_flow_impact as field name in response)

    - Just include the interger values in current payment term and potential cash flow impact.
    - There will be a list name "suppliers" which will contain all the supplier data with the field names as mentioned.

    Example JSON: {example_json}
    Strictly follow the above format and keep the same field names in response everywhere.
    Do not change any values from the input data.

    Return the response in JSON format.
    """
    )
    response = groq_llm.invoke(prompt,response_format={"type": "json_object"})
    # response = groq_llm(message=prompt,
    #     max_tokens=4096,
    #     response_format={ "type": "json_object" }
    # )
    # return response.text
    result2 = response.content

    # st.write(result2)

    def process_supplier_data(data):
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        for supplier in data['suppliers']:
            current_term = supplier['current_payment_term_days']
            negotiation_range = supplier['negotiation_range']
            
            if negotiation_range == 'High':
                supplier['recommended_new_payment_term_days'] = 'Nil'
                supplier['potential_cash_flow_impact'] = 0
            elif negotiation_range == 'Medium':
                if current_term < 60:
                    supplier['recommended_new_payment_term_days'] = current_term + 10
                else:
                    supplier['recommended_new_payment_term_days'] = 'Nil'
                    supplier['potential_cash_flow_impact'] = 0
            elif negotiation_range == 'Low':
                if current_term < 60:
                    supplier['recommended_new_payment_term_days'] = current_term + 15
                elif current_term < 90:
                    supplier['recommended_new_payment_term_days'] = current_term + 5
                else:
                    supplier['recommended_new_payment_term_days'] = 'Nil'
                    supplier['potential_cash_flow_impact'] = 0

        return data

    def create_markdown_table(data):
        headers = [
            "Supplier Name",
            "Total Spend($)",
            "Current Payment Term (days)",
            # "Negotiation Range",
            "Recommended New Payment Term (days)",
            "Potential Cash Flow Impact",
            # "Financial Score"
        ]
        
        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---" for _ in headers]) + " |\n"
        
        for supplier in data['suppliers']:
            row = [
                supplier['supplier_name'],
                f"${supplier['total_spend($)']:,}",
                str(supplier['current_payment_term_days']),
                # supplier['negotiation_range'],
                str(supplier['recommended_new_payment_term_days']),
                f"${supplier['potential_cash_flow_impact']:,}",
                # f"{supplier['financial_score']:.2f}"
            ]
            markdown += "| " + " | ".join(row) + " |\n"
        
        return markdown
    
    processed_data = process_supplier_data(result2)

    # st.write(processed_data)
    
    # st.write(processed_data)
    markdown_table = create_markdown_table(processed_data)

    # def display_negotiation_summary(processed_data):
    #     st.write("**Payment Term Negotiation Summary:**")
    #     recommendations = []
    #     for supplier in processed_data['suppliers']:
    #         if supplier['recommended_new_payment_term_days'] != 'Nil':
    #             message = (f"For {supplier['supplier_name']} the new recommended payment term would be "
    #                     f"{supplier['recommended_new_payment_term_days']} days which will lead to the "
    #                     f"potential cash flow impact of ${supplier['potential_cash_flow_impact']}.")
    #             recommendations.append(message)
        
    #     if recommendations:
    #         for recommendation in recommendations:
    #             st.write(recommendation)
    #     else:
    #         st.write("No recommendations to display.")



    # prompt = f"""
    # Supplier Payment Terms Negotiation Analysis
                
    # (Background - You are an AI assistant tasked with analyzing supplier payment terms and providing specific negotiation recommendations. Your goal is to suggest new payment terms for each supplier and calculate the potential cash flow impact based on the given data.)

    #     Input data: {markdown_table}(The input data is in tabular form)
    #     The input data contains a tab
        
    #     Please provide a brief overall summary highlighting:(under the subheading - Summary and Recommendations:)
        
    #     Summary:
    #         - brief summary in bullet points.

    #     Payment Term Negotiation - 
    #         - Payment term negotiation recommendations based on the input data.
    #         - Provide for all the suppliers.

    #     Don't include any other text in the response.
        

    #     ## Important Notes
    #     - if Recommended payment term is nil, then payment term negotiation is not recommended for that supplier.
        
    #     - Do not invent or assume any data not present in the DataFrame.

    #     - Carefully access all the Input Data before processing.
        
    #     - Only display summary and payment term negotiation recommendations in the result.
        
    #     - Round all financial figures to the nearest dollar.

    #     - Do not display any rules or logic in your response from the prompt.

    #     - Keep the same text format throughout.
        
    #     ## Do not include code snippets"""
 
    # chat_completion = tclien.chat.completions.create(
    #     messages=[{"role": "user", "content": prompt}],
    #     model="llama-3.1-70b-versatile",
    # )
 
    # result = chat_completion.choices[0].message.content

    st.subheader("3. Payment terms recommendations")
    st.write(markdown_table)
    # st.markdown(display_negotiation_summary(processed_data))
    st.write("**Payment Term Negotiation Summary:**")
    for supplier in processed_data['suppliers']:
        if supplier['recommended_new_payment_term_days'] != 'Nil':
            message = (f"For {supplier['supplier_name']} the new recommended payment term would be "
                       f"{supplier['recommended_new_payment_term_days']} days which will lead to the "
                       f"potential cash flow impact of ${supplier['potential_cash_flow_impact']}.")
            st.write(message)

def process_and_build_output(suppliers_df, removed_supplier_total_spend):
    # Convert DataFrame to list of dictionaries for easy processing
    suppliers = suppliers_df.to_dict(orient='records')
   
    # List of contracted suppliers
    contracted_suppliers = [s for s in suppliers if s['Contract Status'] == 'Yes']
   
    # Consolidated suppliers list
    consolidated_suppliers = ', '.join(suppliers_df['Supplier Name'].unique())
    contracted_supplier_names = ', '.join(suppliers_df.loc[suppliers_df['Contract Status'] == 'Yes', 'Supplier Name'].unique()) or 'None'
    non_contracted_supplier_names = ', '.join(suppliers_df.loc[suppliers_df['Contract Status'] == 'No', 'Supplier Name'].unique()) or 'None'

    result = {}
   
    if contracted_suppliers:
        # Find the best supplier based on score, unit price, total spend, and contract value
        min_score_supplier = min(contracted_suppliers, key=lambda x: (
            -x['Supplier Performance Score'],
            x['Average Unit Price($)'],
            x['Total Spend($)'],
            x['Contract Total Value']
        ))
       
        # Extract the total spend from the removed supplier
        if isinstance(removed_supplier_total_spend, pd.DataFrame):
            # Assuming there's only one row and a 'Total Spend' column in the DataFrame
            if 'Total Spend($)' in removed_supplier_total_spend.columns:
                removed_supplier_total_spend = removed_supplier_total_spend['Total Spend($)'].sum()
            else:
                st.error("No 'Total Spend' column found in removed_supplier_total_spend DataFrame")
                return
        else:
            try:
                removed_supplier_total_spend = float(removed_supplier_total_spend)
            except ValueError:
                st.error("Invalid value for 'removed_supplier_total_spend'. Please provide a numeric value.")
                return

        # Ensure 'Contract Total Value' and 'removed_supplier_total_spend' are numeric
        contract_total_value = pd.to_numeric(min_score_supplier['Contract Total Value'], errors='coerce')

        # Calculate the proposed contract value
        if pd.notna(contract_total_value):
            min_score_supplier['Proposed Contract Value'] = contract_total_value + removed_supplier_total_spend
        else:
            min_score_supplier['Proposed Contract Value'] = "Invalid data for calculation"

        # Store the best supplier's information
        result['Best Supplier for Contract Negotiation'] = {
            'Supplier Name': min_score_supplier['Supplier Name'],
            'Contract ID': min_score_supplier['Contract ID'],
            'Current Contract Value': min_score_supplier['Contract Total Value'],
            'Proposed Contract Value': min_score_supplier['Proposed Contract Value'],
            'Supplier Performance Score': min_score_supplier['Supplier Performance Score'],
            'Average Unit Price($)': min_score_supplier['Average Unit Price($)'],
            'Total Spend($)': min_score_supplier['Total Spend($)']
        }
   
    # Non-contracted suppliers list for potential new contracts
    non_contracted_suppliers = [s['Supplier Name'] for s in suppliers if s['Contract Status'] == 'No']
    if non_contracted_suppliers:
        result['Non-contracted Suppliers for New Contracts'] = non_contracted_suppliers
   
    # Add consolidated supplier details
    result['Consolidated Suppliers'] = consolidated_suppliers
    result['Suppliers with Contracts'] = contracted_supplier_names
    result['Suppliers without Contracts'] = non_contracted_supplier_names
   
    st.subheader("4. Contract Term Negotiation")
    output = f"""
 
    1. **Consolidated Suppliers for the Product**:
       - The consolidated suppliers for this product are: {result['Consolidated Suppliers']}.
 
    2. **Contract Status**:
       - Suppliers with contracts: {result['Suppliers with Contracts']}.
       - Suppliers without contracts: {result['Suppliers without Contracts']}.
    """
 
    # Add best supplier for contract renegotiation
    if 'Best Supplier for Contract Negotiation' in result:
        best_supplier = result['Best Supplier for Contract Negotiation']
        output += f"""
    3. **Best Supplier for Contract Renegotiation**:
       - The best supplier to renegotiate a contract with is {best_supplier['Supplier Name']}, and here are the key factors that contribute to this decision:
         - **Supplier Performance Score**: {format_with_commas(best_supplier['Supplier Performance Score'])} (the highest score among all contracted suppliers, indicating strong reliability and overall performance).
         - **Unit Price**: \${format_with_commas(best_supplier['Average Unit Price($)'])} (the lowest unit price among all contracted suppliers, making them a cost-effective option).
         - **Total Spend**: \${format_with_commas(best_supplier['Total Spend($)'])} (this supplier has a relatively lower total spend, making them a potential candidate for further negotiation or increased contract value).
         - **Contract ID** and **Current Contract Value**: The existing contract ({best_supplier['Contract ID']}) has a current value of {format_with_commas(best_supplier['Current Contract Value'])}, which is proposed to increase to {format_with_commas(best_supplier['Proposed Contract Value'])}. This increase accounts for the removed supplier's total spend of \${format_with_commas(removed_supplier_total_spend)} and reflects the supplier’s strong performance. 
          """
   
    # Add non-contracted suppliers for new contract negotiation
    if 'Non-contracted Suppliers for New Contracts' in result:
        output += f"""
    4. **New Contract Proposal**:
       - We propose negotiating new contracts with the non-contracted suppliers: {', '.join(result['Non-contracted Suppliers for New Contracts'])}.
        """
    else:
        output += """
    4. **New Contract Proposal**:
        - **Supplier Availability**:  
            Two non-contracted suppliers, Turner and Allen & Roberts-Ferguson, have been identified as potential candidates for new contract negotiations. We recommend initiating contract discussions with these suppliers to expand our vendor base and potentially secure more favorable terms.  

         
        """


   
    # Write the output to the Streamlit app
    st.markdown(output)
 

# df1 = pd.read_csv("D:\AI_team\streamlit-tail-spend-analysis\main_again.csv")
# df2 = pd.read_csv("D:\AI_team\streamlit-tail-spend-analysis\Tail_again.csv")

# df = process_supplier_data(df1, df2)
# consol_sup, elim_sup = analyze_and_display_suppliers(df, threshold=0.5)
# analyze_and_display_suppliers(df, threshold=0.5)
# display_negotiation_recommendations(consol_sup)
# negotiate_payment_terms(consol_sup)
# process_and_build_output(consol_sup, elim_sup)
