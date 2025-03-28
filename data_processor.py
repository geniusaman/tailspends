import pandas as pd
from data_loader import load_data

# Function to preprocess the data
def preprocess_data(data):
    # Convert date columns to datetime
    # date_columns = ['contract_start_date', 'contract_end_date', 'purchase_date', 'delivery_date']
    # for col in date_columns:
    #     data[col] = pd.to_datetime(data[col])
    
    # Extract time-related fields
    data['quarter'] = data['po_date'].dt.to_period('Q').astype(str)
    data['year'] = data['po_date'].dt.year
    data['month'] = data['po_date'].dt.to_period('M').astype(str)
    
    return data

# Function to filter data based on various criteria
def filter_data(df, selected_category='All', selected_subcategory='All', year='All', month='All', quarter='All', country='All'):
    # Apply category and subcategory filters
    if selected_category != 'All':
        df = df[df['parent_category'] == selected_category]
    if selected_subcategory != 'All':
        df = df[df['category'] == selected_subcategory]
    
    # Apply time-related filters
    if year != 'All':
        df = df[df['year'] == int(year)]
    if month != 'All':
        df = df[df['month'] == month]
    if quarter != 'All':
        df = df[df['quarter'] == quarter]
    
    # Apply country filter
    if country != 'All':
        df = df[df['country'] == country]
    
    return df

# def classify_suppliers(df, threshold=0.8):
#     # Step 1: Calculate total spend for each supplier
#     supplier_spend = df.groupby('supplier_name')['totalcost'].sum().reset_index()
#     supplier_spend.columns = ['supplier_name', 'Total Spend']

#     # Step 2: Order suppliers from highest to lowest total spend
#     supplier_spend = supplier_spend.sort_values(by='Total Spend', ascending=False)

#     # Step 3: Calculate cumulative spend and cutoff point (80% of total spend)
#     total_spend = supplier_spend['Total Spend'].sum()
#     supplier_spend['Cumulative Spend'] = supplier_spend['Total Spend'].cumsum()
#     supplier_spend['Cutoff'] = total_spend * threshold

#     # Step 4: Identify main suppliers and tail spend suppliers
#     main_suppliers = supplier_spend[supplier_spend['Cumulative Spend'] <= supplier_spend['Cutoff']]
#     tail_suppliers = supplier_spend[supplier_spend['Cumulative Spend'] > supplier_spend['Cutoff']]

#     # Step 5: Calculate total tail spend
#     total_tail_spend = tail_suppliers['Total Spend'].sum()

#     # Step 6: Filter the original data to get all the columns for main and tail suppliers
#     main_suppliers_grouped = df[df['supplier_name'].isin(main_suppliers['supplier_name'])]
#     tail_suppliers_grouped = df[df['supplier_name'].isin(tail_suppliers['supplier_name'])]

#     return main_suppliers_grouped, tail_suppliers_grouped, total_tail_spend

def classify_suppliers(df, threshold=0.8):
    # Step 1: Calculate total spend for each supplier
    supplier_spend = df.groupby('supplier_name')['po_amount'].sum().reset_index()
    supplier_spend.columns = ['supplier_name', 'Total Spend']

    # Step 2: Order suppliers from highest to lowest total spend
    supplier_spend = supplier_spend.sort_values(by='Total Spend', ascending=False)

    # Step 3: Calculate cumulative spend and cutoff point
    supplier_spend['Cumulative Spend'] = supplier_spend['Total Spend'].cumsum()
    total_spend = supplier_spend['Total Spend'].sum()
    cutoff_index = supplier_spend[supplier_spend['Cumulative Spend'] / total_spend >= threshold].index[0]
    supplier_spend['Cutoff'] = supplier_spend.loc[cutoff_index, 'Cumulative Spend']

    # Step 4: Identify main suppliers and tail spend suppliers
    main_suppliers = supplier_spend[supplier_spend['Cumulative Spend'] <= supplier_spend['Cutoff']]
    tail_suppliers = supplier_spend[supplier_spend['Cumulative Spend'] > supplier_spend['Cutoff']]

    # Step 5: Calculate total tail spend
    total_tail_spend = tail_suppliers['Total Spend'].sum()

    # Step 6: Filter the original data to get all the columns for main and tail suppliers
    main_suppliers_grouped = df[df['supplier_name'].isin(main_suppliers['supplier_name'])]
    tail_suppliers_grouped = df[df['supplier_name'].isin(tail_suppliers['supplier_name'])]

    return main_suppliers_grouped, tail_suppliers_grouped, total_tail_spend

# Example Usage
def run_analysis(df, selected_category='All', selected_subcategory='All', year='All', month='All', quarter='All', country='All'):
    # Step 1: Apply filters
    filtered_data = filter_data(df, selected_category, selected_subcategory, year, month, quarter, country)

    # Step 2: Classify suppliers
    main_suppliers_grouped, tail_suppliers_grouped, total_tail_spend = classify_suppliers(filtered_data)

    return main_suppliers_grouped, tail_suppliers_grouped, total_tail_spend

data = load_data()
print(data.columns)
# Load and preprocess data
# Filter for Stainless Steel category
data = data[data['category'] == 'Stainless Steel']
data = preprocess_data(data)

# Run the analysis with additional filters
main_suppliers, tail_suppliers, total_tail_spend = run_analysis(data, selected_category='All', selected_subcategory='All', year='All', month='All', quarter='All', country='All')

# Display the results
# print("Main Suppliers:\n\n", main_suppliers)
# print("\n\nTail Suppliers:\n\n", tail_suppliers)
# print("\n\nTotal Tail Spend:", total_tail_spend)
