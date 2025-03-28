import os
import base64
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data_processor import preprocess_data, classify_suppliers
from utils import create_kpi_metric
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
from PIL import Image
import pandas as pd
from data_loader import load_data,load_contract_data
import streamlit as st
import os
from groq import Groq
from utils import create_kpi_metric
from recommendation import process_supplier_data, analyze_and_display_suppliers, display_negotiation_recommendations, negotiate_payment_terms, process_and_build_output

# st.set_page_config(
#     page_title="Cost Saving Recommendation",  
#     page_icon="favicon.ico",    
#     layout="wide"
# )

main_color = '#5e17eb'
secondary_color = '#a17ffa'


st.markdown("""
<style>
    html, body, section{
        overflow-x: hidden;
    }

    .stApp {
        background-color: #fff;
    }
    .dashboard-section {
        background-color: #ddd;
        padding: 1px;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
        margin-bottom: 1rem;
    }
    .dashboard-section h3 {
        color: #36b9cc;
        margin-bottom: 0.5rem;
    }
    .kpi-section {
        background-color: #ddd;
        width: 1px;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .kpi-metric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid rgba(0, 0, 0, 0.125);
    }
    .stMetric {
        background-color: #F3F2FF;
        padding: 2rem 0 2rem 2rem;
        border-radius: 1rem;
    
    }
            
    .stMetric > div {
        margin-top: 0.5rem;
        font-weight: bold;        
    }
            
    # .st-emotion-cache-c9mv52 .e1f1d6gn2, .st-emotion-cache-6fh3fb .e1f1d6gn2, .st-emotion-cache-1n76uvr .e1f1d6gn2, .st-emotion-cache-ir0yj4 .e1f1d6gn2{
    #     background-color: #F3F3F6;
    #     border-radius: 1rem;
    #     padding: 1rem;
    # }
            
    # #general-spend-analysis{
    #     font-size: 1.2rem;
    # }

    .stSlider, .stSelectbox {
        max-width: 96%;
    }
    .stExpander {
        border: 1px solid #5e17eb;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stExpander > div:first-child {
        border-bottom: 1px solid #5e17eb;
        font-weight: 600;
        color: #5e17eb;
    }
            
    button p, [data-testid="stMarkdownContainer"]{
        width: 100%;
    }
    
    p{
        width: 95%;
    }
            
    [data-testid="stMetricLabel"] p{
            width: 100%;
    }
            
    [data-testid="stSidebarContent"]{
        border-right: 1px solid #ddd;        
    }
            
    .st-emotion-cache-0 .e1f1d6gn0 > .st-emotion-cache-1wmy9hl .e1f1d6gn1{
        padding: 1rem;
        border: 0px solid #ddd;
        border-radius: 0.5rem;
        background-color: #f3f2ff;
    }
            
    [data-baseweb="checkbox"] p{
        width: 100%;        
    }
    
    [data-testid="stTabs"]{
        border: 0px solid #ddd;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.1);        
        width: 100%;
        padding: 1rem;
    }
            
     [data-baseweb="tab-panel"]  .st-emotion-cache-0 .e1f1d6gn0{
        width: 96%;
    }
            
    [data-baseweb="select"], .stSelectbox > div{
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }
    header{
        opacity: 0;
        border-bottom: 1px solid #ddd;
    }        
</style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "logo.png")

    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        logo_html = f"""
        <div style="position: absolute; top: -6rem; left: 10px; ">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 10rem; z-index: 1000 !important;">
            
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.error(f"Logo file not found at {logo_path}")


    # st.markdown("""
    # <hr style="border: .01rem solid #ddd; width: 100vw; position: absolute; top: -6rem; left: -4.8rem;">
    # """, unsafe_allow_html=True)

    st.markdown('<h1 style="text-align: center;">Spend Analysis</h1>', unsafe_allow_html=True)
    data = load_data()

    # Load and preprocess data
    # Filter for Stainless Steel category
    data = data[data['category'] == 'Stainless Steel']
    data = preprocess_data(data)
     # Sidebar filters
    filtered_data = apply_filters(data)
    

    contract_data = load_contract_data()

   

    # Apply spend classification (getting main, tail, and total tail spend) after filtering
    main_suppliers, tail_suppliers, total_tail_spend = classify_suppliers(filtered_data)

    # # KPI Section
    display_kpis(main_suppliers, tail_suppliers, total_tail_spend)


    with st.container():
            st.header('General Spend Analysis')
            general_spend_analysis(main_suppliers, tail_suppliers)

    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            tail_spend_analysis(filtered_data)

    with col2:
        with st.container():
            st.header('Spend Analysis by Subcategory')
            subcategory_analysis(filtered_data)
            
    with st.container():
        st.header('Spend Analysis by Products')
        product_analysis(filtered_data)
    
    with st.container():
        supplier_performance_analysis(filtered_data,contract_data)
        # tail_metrics, main_metrics = supplier_performance_analysis(filtered_data)
        # create_recommendation_button(tail_metrics, main_metrics)

   
    # Add sidebar information
    st.sidebar.markdown("""## About
    """)
    st.sidebar.markdown("This procurement dashboard provides insights into spend analysis, supplier performance, and consolidation opportunities.")
def apply_filters(data):
    st.sidebar.header('Filters')
    
    # Filter by Year
    years = sorted(data['year'].unique(), reverse=True)
    selected_year = st.sidebar.selectbox('Select Year', ['All'] + years)
    
    # Filter by Quarter
    quarters = sorted(data['quarter'].unique()) if selected_year == 'All' else sorted(data[data['year'] == selected_year]['quarter'].unique())
    selected_quarter = st.sidebar.selectbox('Select Quarter', ['All'] + quarters)
    
    # Filter by Month
    months = sorted(data['month'].unique())
    if selected_year != 'All':
        months = sorted(data[data['year'] == selected_year]['month'].unique())
        if selected_quarter != 'All':
            months = sorted(data[(data['year'] == selected_year) & (data['quarter'] == selected_quarter)]['month'].unique())
    selected_month = st.sidebar.selectbox('Select Month', ['All'] + months)
    
    # Filter by Country
    selected_country = st.sidebar.multiselect('Select Country', ['All'] + list(data['country'].unique()))
    
    
    # Apply filters
    filtered_data = data.copy()
    
    if selected_year != 'All':
        filtered_data = filtered_data[filtered_data['year'] == selected_year]
    if selected_quarter != 'All':
        filtered_data = filtered_data[filtered_data['quarter'] == selected_quarter]
    if selected_month != 'All':
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    if selected_country and 'All' not in selected_country:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_country)]
    
    
    return filtered_data

def display_kpis(main_suppliers, tail_suppliers, total_tail_spend):
    st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
    st.header('Key Performance Indicators')
    col1, col2, col3 = st.columns(3)
# Total Spend (Main + Tail Suppliers)
    total_spend = main_suppliers['po_amount'].sum() + tail_suppliers['po_amount'].sum()
    total_spend_million = total_spend / 1_000_000
    with col1:
        create_kpi_metric("Total Spend", f"${total_spend_million:,.2f}M")

    # # Average Delivery Time (Main + Tail Suppliers)
    # avg_delivery_time_main = main_suppliers['avg_delivery_time'].mean()
    # avg_delivery_time_tail = tail_suppliers['avg_delivery_time'].mean()
    # overall_avg_delivery_time = (avg_delivery_time_main + avg_delivery_time_tail) / 2
    # with col2:
    #     create_kpi_metric("Avg. Delivery Time", f"{overall_avg_delivery_time:.0f} days")

    # Tail Spend Ratio (Tail Spend / Total Spend)
    tail_spend_ratio = (total_tail_spend / total_spend) * 100
    with col2:
        create_kpi_metric("Tail Spend Ratio", f"{tail_spend_ratio:.2f}%")

    # Total Suppliers (Main + Tail Suppliers)
    total_suppliers = pd.concat([main_suppliers, tail_suppliers])['supplier_name'].nunique()
    with col3:
        create_kpi_metric("Total Suppliers", total_suppliers)

def general_spend_analysis(main_suppliers, tail_suppliers):
    col1, col2 = st.columns(2)

    # Concatenate main and tail suppliers for overall filtered data
    filtered_data = pd.concat([main_suppliers.assign(spend_type='Main Spend'), tail_suppliers.assign(spend_type='Tail Spend')])

    # Country Spend Analysis
    with col1:
        country_spend = filtered_data.groupby(['country', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()
        country_spend = country_spend.sort_values('Tail Spend', ascending=False)
        fig_country = px.bar(country_spend, x='country', y=['Tail Spend', 'Main Spend'],
                            title='Country Spend', labels={'value': 'Total Spend'}, barmode='stack',
                            color_discrete_map={'Main Spend': secondary_color, 'Tail Spend': main_color})
        fig_country.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_country, use_container_width=True)

    # Category Spend Analysis
    with col2:
        
            # After filtering, classify suppliers as main and tail
        main_suppliers, tail_suppliers, total_tail_spend = classify_suppliers(filtered_data)

        # Concatenate the main and tail supplier data with spend_type column
        filtered_data = pd.concat([main_suppliers.assign(spend_type='Main Spend'), tail_suppliers.assign(spend_type='Tail Spend')])

        # Now group by category and spend type for analysis
        spend_by_category = filtered_data.groupby(['parent_category', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()
        category_spend = filtered_data.groupby(['parent_category', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()
        category_spend = category_spend.sort_values('Tail Spend', ascending=False)
        fig_category = px.bar(category_spend, x='parent_category', y=['Tail Spend', 'Main Spend'],
                            title='Category Spend', labels={'value': 'Total Spend'}, barmode='stack',
                            color_discrete_map={'Main Spend': secondary_color, 'Tail Spend': main_color})
        fig_category.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_category, use_container_width=True)

    col3, col4 = st.columns(2)

    # Supplier Spend Analysis
    with col3:
        supplier_spend = filtered_data.groupby(['supplier_name', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()

        # Ensure 'Main Spend' and 'Tail Spend' columns exist in case they are missing
        if 'Main Spend' not in supplier_spend.columns:
            supplier_spend['Main Spend'] = 0
        if 'Tail Spend' not in supplier_spend.columns:
            supplier_spend['Tail Spend'] = 0

        # Sort data: Main Spend first, then Tail Spend in descending order
        supplier_spend = supplier_spend.sort_values(by=['Main Spend', 'Tail Spend'], ascending=[False, False])

        # Create the bar chart with Main Spend first, then Tail Spend
        fig_supplier = px.bar(supplier_spend, x='supplier_name', y=['Tail Spend', 'Main Spend'],
                            title='Supplier Spend', labels={'value': 'Total Spend'}, barmode='stack',
                            color_discrete_map={'Main Spend': secondary_color, 'Tail Spend': main_color})
        
        fig_supplier.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_supplier, use_container_width=True)



    # Payment Terms Analysis
    with col4:
        payment_terms_data = filtered_data.groupby('payment_terms')['supplier_name'].nunique().reset_index(name='supplier_count')
        payment_terms_data = payment_terms_data.sort_values('supplier_count', ascending=False)

        fig_payment_terms = px.bar(payment_terms_data, x='payment_terms', y='supplier_count',
                                title='Number of Suppliers by Payment Terms',
                                labels={'payment_terms': 'Payment Terms', 'supplier_count': 'Number of Suppliers'})
        fig_payment_terms.update_layout(height=400)
        fig_payment_terms.update_traces(marker_color='#B5B0EF')

        supplier_lists = []
        for term in payment_terms_data['payment_terms']:
            suppliers = filtered_data[filtered_data['payment_terms'] == term]['supplier_name'].unique()
            supplier_list = '<br>'.join(suppliers)
            supplier_lists.append(supplier_list)
        fig_payment_terms.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=40, t=60, b=40)
            
        )
        fig_payment_terms.update_traces(
            hovertemplate='Payment Terms: %{x}<br>Number of Suppliers: %{y}<br>Suppliers:<br>%{customdata}',
            customdata=supplier_lists
        )

        all_suppliers = filtered_data['supplier_name'].unique()
        buttons = [dict(label='All Suppliers', method='update', args=[{'x': [payment_terms_data['payment_terms']],
                                                                    'y': [payment_terms_data['supplier_count']],
                                                                    'customdata': [supplier_lists]}])]

        for supplier in all_suppliers:
            filtered_supplier_data = filtered_data[filtered_data['supplier_name'] == supplier]
            filtered_payment_terms_data = filtered_supplier_data.groupby('payment_terms')['supplier_name'].nunique().reset_index(name='supplier_count')
            filtered_supplier_list = ['<br>'.join(filtered_supplier_data['supplier_name'].unique())]

            if not filtered_payment_terms_data.empty:
                buttons.append(dict(label=supplier, method='update', args=[{'x': [filtered_payment_terms_data['payment_terms']],
                                                                            'y': [filtered_payment_terms_data['supplier_count']],
                                                                            'customdata': [filtered_supplier_list]}]))

        fig_payment_terms.update_layout(
            updatemenus=[dict(
                type='dropdown',
                showactive=True,
                buttons=buttons,
                x=1,
                y=1,
                xanchor='left',
                yanchor='top',
                
            )]
        )

        st.plotly_chart(fig_payment_terms, use_container_width=True)

    # with st.container():
    #     quarterly_discount = filtered_data.groupby(['quarter', 'supplier_name'])['discount'].sum().reset_index()
    #     quarterly_discount = quarterly_discount.sort_values(['quarter', 'discount'], ascending=[True, False])

    #     fig_quarterly_discount = px.bar(quarterly_discount, x='quarter', y='discount', color='supplier_name',
    #                                     title='Quarterly Discount Amount by Supplier',
    #                                     labels={'quarter': 'Quarter', 'discount': 'Discount Amount', 'supplier_name': 'Supplier'})
    #     fig_quarterly_discount.update_layout(height=500, showlegend=False)

    #     buttons = [dict(label='All Suppliers', method='update', args=[{'visible': [True] * len(fig_quarterly_discount.data)}])]

    #     for i, supplier in enumerate(quarterly_discount['supplier_name'].unique()):
    #         visibility = [False] * len(fig_quarterly_discount.data)
    #         visibility[i] = True
    #         buttons.append(dict(label=supplier, method='update', args=[{'visible': visibility}]))

    #     fig_quarterly_discount.update_layout(
    #         plot_bgcolor='rgba(0,0,0,0)',
    #         paper_bgcolor='rgba(0,0,0,0)',
    #         height=500,
    #         margin=dict(l=40, r=40, t=60, b=40),
    #         updatemenus=[dict(
    #             type='dropdown',
    #             showactive=True,
    #             buttons=buttons,
    #             x=.7,
    #             y=1.1,
    #             xanchor='left',
    #             yanchor='top'
    #         )]
    #     )

    #     st.plotly_chart(fig_quarterly_discount, use_container_width=True)

def tail_spend_analysis(filtered_data):
    st.header('Spend Analysis by Category')

    # Dropdown for selecting a specific category for detailed analysis
    spend_by_category = filtered_data.groupby('parent_category')['po_amount'].sum().reset_index()
    selected_category = st.selectbox('Select Category for Detailed Analysis', spend_by_category['parent_category'].unique(), key='tail_spend_category_selectbox')

    if selected_category and 'All' not in selected_category:
        # Apply the filter for the selected category first
        filtered_data = filtered_data[filtered_data['parent_category'].isin([selected_category])]

    # After filtering, classify suppliers as main and tail
    main_suppliers, tail_suppliers, total_tail_spend = classify_suppliers(filtered_data)

    # Concatenate the main and tail supplier data with spend_type column
    filtered_data = pd.concat([main_suppliers.assign(spend_type='Main Spend'), tail_suppliers.assign(spend_type='Tail Spend')])

    # Now group by category and spend type for analysis
    spend_by_category = filtered_data.groupby(['parent_category', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()

    # Sort the data by 'Tail Spend' in descending order, safeguard for missing columns
    if 'Tail Spend' in spend_by_category.columns:
        spend_by_category = spend_by_category.sort_values('Tail Spend', ascending=False)

    if selected_category:
        st.session_state.selected_category = selected_category

        # Filter the data for the selected category (again, after classifying suppliers)
        category_data = filtered_data[filtered_data['parent_category'] == selected_category]
        subcategory_spend = category_data.groupby(['parent_category', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()

        # Ensure both 'Main Spend' and 'Tail Spend' columns are present, even if empty
        if 'Tail Spend' not in subcategory_spend.columns:
            subcategory_spend['Tail Spend'] = 0
        if 'Main Spend' not in subcategory_spend.columns:
            subcategory_spend['Main Spend'] = 0

        # Sort the subcategory data by 'Tail Spend'
        subcategory_spend = subcategory_spend.sort_values('Tail Spend', ascending=False)

        # Create a stacked bar chart for subcategory spend analysis
        fig_subcategory = px.bar(subcategory_spend, x='parent_category', y=['Tail Spend', 'Main Spend'], 
                                title=f'Spend by Subcategory in {selected_category}', 
                                labels={'value': 'Spend'},
                                barmode='stack',
                                color_discrete_map={'Main Spend': secondary_color, 'Tail Spend': main_color})
        fig_subcategory.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=724,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
        )
        st.plotly_chart(fig_subcategory, use_container_width=True)


def subcategory_analysis(filtered_data):
    # Ensure a category is selected first
    if 'selected_category' not in st.session_state:
        st.warning("Please select a category first in Tail Spend Analysis.")
        return

    # Get the selected category from session state
    selected_category = st.session_state.selected_category

    # Filter the data by selected category
    category_data = filtered_data[filtered_data['parent_category'] == selected_category]

    # Dropdown for selecting a specific subcategory
    selected_subcategory = st.selectbox('Select Subcategory for Detailed Analysis',
                                        category_data['category'].unique(),
                                        key='subcategory_selectbox')

    # Apply subcategory filter first
    if selected_subcategory:
        # Save the selected subcategory in session state
        st.session_state.selected_subcategory = selected_subcategory

        # Filter data based on the selected subcategory
        subcategory_data = category_data[category_data['category'] == selected_subcategory]

        # After filtering by subcategory, classify suppliers
        main_suppliers, tail_suppliers, total_tail_spend = classify_suppliers(subcategory_data)

        # Concatenate the classified data (main and tail suppliers) with spend_type
        classified_data = pd.concat([main_suppliers.assign(spend_type='Main Spend'),
                                     tail_suppliers.assign(spend_type='Tail Spend')])

        # If no data exists for Main or Tail Spend, show a warning
        if classified_data.empty:
            st.warning(f"No spend data available for {selected_subcategory}.")
            return

    
        tab1, tab2 = st.tabs(["Purchase Frequency", "Supplier Spend"])
        
        with tab1:
            # Group by supplier_name and spend_type, then count the frequency of purchases
            supplier_frequency = classified_data.groupby(['supplier_name', 'spend_type']).size().reset_index(name='frequency')

            # Sort the values by 'frequency' in descending order
            supplier_frequency = supplier_frequency.sort_values(by='frequency', ascending=False)

            # Create the bar chart for supplier frequency
            if not supplier_frequency.empty:
                fig_supplier_freq = px.bar(
                    supplier_frequency,
                    x='supplier_name',
                    y='frequency',
                    color='spend_type',
                    title=f'Suppliers by Purchase Frequency in {selected_subcategory}',
                    labels={'frequency': 'Purchase Frequency', 'supplier_name': 'Supplier'},
                    barmode='stack',
                    color_discrete_map={'Main Spend': main_color, 'Tail Spend': secondary_color},
                    category_orders={'supplier_name': supplier_frequency['supplier_name'].tolist()}
                )
                fig_supplier_freq.update_layout(  plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=602,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
                st.plotly_chart(fig_supplier_freq, use_container_width=True)
            else:
                st.warning(f"No purchase frequency data available for {selected_subcategory}.")

        with tab2:
            # Group by supplier_name and spend_type, then sum the 'totalcost'
            supplier_spend = classified_data.groupby(['supplier_name', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()

            # Check for existence of main and tail spend
            has_main_spend = 'Main Spend' in supplier_spend.columns and supplier_spend['Main Spend'].sum() > 0
            has_tail_spend = 'Tail Spend' in supplier_spend.columns and supplier_spend['Tail Spend'].sum() > 0

            if has_main_spend or has_tail_spend:
                # Create sliders for Main and Tail Spend
                if has_main_spend:
                    min_main_spend = supplier_spend['Main Spend'].min()
                    max_main_spend = supplier_spend['Main Spend'].max()
                    if min_main_spend < max_main_spend:
                        main_spend_range = st.slider(
                            "Main Spend Range", 
                            min_value=float(min_main_spend),
                            max_value=float(max_main_spend),
                            value=(float(min_main_spend), float(max_main_spend)),
                        )
                    else:
                        st.warning("Only one value exists for Main Spend; slider disabled.")
                        main_spend_range = (min_main_spend, max_main_spend)

                if has_tail_spend:
                    min_tail_spend = supplier_spend['Tail Spend'].min()
                    max_tail_spend = supplier_spend['Tail Spend'].max()
                    if min_tail_spend < max_tail_spend:
                        tail_spend_range = st.slider(
                            "Tail Spend Range", 
                            min_value=float(min_tail_spend),
                            max_value=float(max_tail_spend),
                            value=(float(min_tail_spend), float(max_tail_spend)),
                        )
                    else:
                        # st.warning("Only one value exists for Tail Spend; slider disabled.")
                        tail_spend_range = (min_tail_spend, max_tail_spend)

                # Filter based on selected ranges
                filtered_supplier_spend = supplier_spend[
                    (has_main_spend and supplier_spend['Main Spend'].between(main_spend_range[0], main_spend_range[1])) |
                    (has_tail_spend and supplier_spend['Tail Spend'].between(tail_spend_range[0], tail_spend_range[1]))
                ]

                # Check if filtered_supplier_spend has data
                if not filtered_supplier_spend.empty:
                    # Create the bar chart to display Main and Tail Spend for each supplier
                    y_values = []
                    if has_tail_spend:
                        y_values.append('Tail Spend')
                    if has_main_spend:
                        y_values.append('Main Spend')

                    fig_supplier_spend = px.bar(
                        filtered_supplier_spend,
                        x='supplier_name',
                        y=y_values,
                        title=f'Supplier Spend in {selected_subcategory}',
                        labels={'value': 'Spend', 'supplier_name': 'Supplier'},
                        barmode='stack',
                        color_discrete_map={'Main Spend': main_color, 'Tail Spend': secondary_color},
                    )
                    fig_supplier_spend.update_layout(  plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=724,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
                    st.plotly_chart(fig_supplier_spend, use_container_width=True)
                else:
                    st.warning(f"No suppliers found within the selected spend range for {selected_subcategory}.")
            else:
                st.warning(f"No spend data available for {selected_subcategory}.")


def product_analysis(filtered_data):
    if 'selected_category' not in st.session_state:
        st.warning("Please select a category first in Tail Spend Analysis.")
        return

    selected_category = st.session_state.selected_category
    category_data = filtered_data[filtered_data['parent_category'] == selected_category]

    # Select subcategory first
    selected_subcategory = st.selectbox('Select Subcategory for Detailed Analysis', category_data['category'].unique(), key='product_subcategory_selectbox')

    if selected_subcategory:
        subcategory_data = category_data[category_data['category'] == selected_subcategory]

        # Select product (item_name) for further analysis
        selected_product = st.selectbox('Select Product for Detailed Analysis', subcategory_data['product_name'].unique(), key='product_selectbox')

        if selected_product:
            # Filter by selected product (item_name)
            product_data = subcategory_data[subcategory_data['product_name'] == selected_product]

            # Classify suppliers for the selected product (item_name)
            main_suppliers, tail_suppliers, total_tail_spend = classify_suppliers(product_data)

            # Concatenate main and tail supplier data after classification
            classified_data = pd.concat([
                main_suppliers.assign(spend_type='Main Spend'),
                tail_suppliers.assign(spend_type='Tail Spend')
            ])

            st.session_state.selected_product = selected_product
            
            tab1, tab2 = st.tabs([f'Product Spend by Supplier for {selected_product}', f'Product Quantity by Supplier for {selected_product}'])

            with tab1:
                # Supplier spend grouped by spend type (Main/Tail Spend)
                supplier_spend_by_product = classified_data.groupby(['supplier_name', 'spend_type'])['po_amount'].sum().unstack(fill_value=0).reset_index()

                # Check for existence of main and tail spend
                has_main_spend = 'Main Spend' in supplier_spend_by_product.columns and supplier_spend_by_product['Main Spend'].sum() > 0
                has_tail_spend = 'Tail Spend' in supplier_spend_by_product.columns and supplier_spend_by_product['Tail Spend'].sum() > 0

                if has_main_spend or has_tail_spend:
                    # Add sliders for Tail Spend and Main Spend ranges
                    if has_tail_spend:
                        tail_spend_range = st.slider("Tail Spend Range", 
                            min_value=float(supplier_spend_by_product['Tail Spend'].min()),
                            max_value=float(supplier_spend_by_product['Tail Spend'].max()),
                            value=(float(supplier_spend_by_product['Tail Spend'].min()), float(supplier_spend_by_product['Tail Spend'].max()) + 1),
                            key='product_tail_spend_slider')
                    else:
                        # st.warning("No Tail Spend data available; slider disabled.")
                        tail_spend_range = (0, 0)

                    if has_main_spend:
                        main_spend_range = st.slider("Main Spend Range", 
                            min_value=float(supplier_spend_by_product['Main Spend'].min()),
                            max_value=float(supplier_spend_by_product['Main Spend'].max()),
                            value=(float(supplier_spend_by_product['Main Spend'].min()), float(supplier_spend_by_product['Main Spend'].max()) + 1),
                            key='product_main_spend_slider')
                    else:
                        # st.warning("No Main Spend data available; slider disabled.")
                        main_spend_range = (0, 0)

                    filtered_supplier_spend_by_product = supplier_spend_by_product[
                (not has_tail_spend or supplier_spend_by_product['Tail Spend'].between(tail_spend_range[0], tail_spend_range[1])) &
                (not has_main_spend or supplier_spend_by_product['Main Spend'].between(main_spend_range[0], main_spend_range[1]))
            ]

                    # Check if filtered data has any suppliers to plot
                    if not filtered_supplier_spend_by_product.empty:
                        # Prepare y values and color map
                        y_values = []
                        color_map = {}
                        if has_tail_spend:
                            y_values.append('Tail Spend')
                            color_map['Tail Spend'] = main_color
                        if has_main_spend:
                            y_values.append('Main Spend')
                            color_map['Main Spend'] = secondary_color

                        if y_values:  # Only create chart if we have y values
                            # Create a bar chart for supplier spend
                            fig_product_supplier_spend = px.bar(filtered_supplier_spend_by_product, 
                                                                x='supplier_name', 
                                                                y=y_values,
                                                                title=f'Product Spend by Supplier for {selected_product}',
                                                                labels={'value': 'Product Spend', 'supplier_name': 'Supplier'},
                                                                barmode='stack',
                                                                color_discrete_map=color_map)

                            fig_product_supplier_spend.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=500,
                                margin=dict(l=40, r=40, t=60, b=40),
                                legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                xaxis_tickangle=-45
                            )

                            st.plotly_chart(fig_product_supplier_spend, use_container_width=True)
                        else:
                            st.warning(f"No spend data available for {selected_product}.")
                    else:
                        st.warning(f"No suppliers found within the selected spend range for {selected_product}.")
                else:
                    st.warning(f"No spend data available for {selected_product}.")
            with tab2:
                # Supplier quantity grouped by spend type (Main/Tail Spend)
                supplier_quantity_by_product = classified_data.groupby(['supplier_name', 'spend_type'])['quantity'].sum().unstack(fill_value=0).reset_index()

                # Check if there are main and tail quantities
                has_main_quantity = 'Main Spend' in supplier_quantity_by_product.columns and supplier_quantity_by_product['Main Spend'].sum() > 0
                has_tail_quantity = 'Tail Spend' in supplier_quantity_by_product.columns and supplier_quantity_by_product['Tail Spend'].sum() > 0

                if has_main_quantity or has_tail_quantity:
                    # Calculate total quantity for sorting
                    supplier_quantity_by_product['Total Quantity'] = 0
                    if has_tail_quantity:
                        supplier_quantity_by_product['Total Quantity'] += supplier_quantity_by_product['Tail Spend']
                    if has_main_quantity:
                        supplier_quantity_by_product['Total Quantity'] += supplier_quantity_by_product['Main Spend']

                    # Sort by Total Quantity in descending order
                    supplier_quantity_by_product = supplier_quantity_by_product.sort_values(by='Total Quantity', ascending=False)

                    # Prepare y values and color map
                    y_values = []
                    color_map = {}
                    if has_tail_quantity:
                        y_values.append('Tail Spend')
                        color_map['Tail Spend'] = main_color
                    if has_main_quantity:
                        y_values.append('Main Spend')
                        color_map['Main Spend'] = secondary_color

                    if y_values:  # Only create chart if we have y values
                        # Create a bar chart for supplier quantity
                        fig_product_supplier_quantity = px.bar(supplier_quantity_by_product, 
                                                                x='supplier_name', 
                                                                y=y_values,
                                                                title=f'Product Quantity by Supplier for {selected_product}',
                                                                labels={'supplier_name': 'Supplier', 'value': 'Quantity'},
                                                                barmode='stack',
                                                                color_discrete_map=color_map)

                        fig_product_supplier_quantity.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=500,
                                margin=dict(l=40, r=40, t=60, b=40),
                                legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig_product_supplier_quantity, use_container_width=True)
                    else:
                        st.warning(f"No quantity data available for {selected_product}.")
                else:
                    st.warning(f"No quantity data available for {selected_product}.")
def supplier_performance_analysis(filtered_data,contract_data):
    st.divider()
    st.header("Supplier Performance")

    # Check if a product is selected in the product analysis
    if 'selected_product' not in st.session_state:
        st.warning("Please select a product first in Tail Spend Analysis.")
        return

    selected_product = st.session_state.selected_product

    # Apply filters to the filtered_data to get relevant product data
    product_data = filtered_data[filtered_data['product_name'] == selected_product]
    # st.write(product_data)

    # Retrieve the tail spend range and main spend range selected in the product analysis
    tail_spend_range = st.session_state.get('product_tail_spend_slider', (0, float('inf')))
    main_spend_range = st.session_state.get('product_main_spend_slider', (0, float('inf')))

    # Classify suppliers based on the filtered product data
    main_suppliers, tail_suppliers, total_tail_spend = classify_suppliers(product_data)

    # Initialize metrics
    tail_supplier_comparison_metrics = None
    main_supplier_comparison_metrics = None

    # Tail Suppliers Analysis
    st.subheader(f"{selected_product} - Tail Suppliers Comparison Analysis")
    tail_only_suppliers = tail_suppliers[
        tail_suppliers['po_amount'].between(tail_spend_range[0], tail_spend_range[1])
    ]['supplier_name'].unique()

    all_option = 'Select All'
    if len(tail_only_suppliers) > 0:
        supplier_options = [all_option] + list(tail_only_suppliers)
        selected_suppliers = st.multiselect(
            'Select Tail Suppliers for Comparison',
            supplier_options,
            key='supplier_comparison'
        )

        if all_option in selected_suppliers:
            # If 'Select All' is chosen, select all available suppliers
            selected_suppliers = tail_only_suppliers.tolist()

        if selected_suppliers:
            supplier_data = product_data[product_data['supplier_name'].isin(selected_suppliers)]
            tail_metrics = LLMdisplay_supplier_comparison(supplier_data,contract_data)
            display_supplier_comparison(supplier_data, '#B5B0EF',contract_data)
        else:
            st.info("Please select tail suppliers to compare.")
    else:
        st.info("No tail suppliers available for comparison.")

    # Main Suppliers Analysis
    st.subheader(f"{selected_product} - Main Suppliers Comparison Analysis")
    main_only_suppliers = main_suppliers[
        main_suppliers['po_amount'].between(main_spend_range[0], main_spend_range[1])
    ]['supplier_name'].unique()

    if len(main_only_suppliers) > 0:
        supplier_options = [all_option] + list(main_only_suppliers)
        selected_suppliers = st.multiselect(
            'Select Main Suppliers for Comparison',
            supplier_options,
            key='main_supplier_comparison'
        )

        if all_option in selected_suppliers:
            # If 'Select All' is chosen, select all available suppliers
            selected_suppliers = main_only_suppliers.tolist()

        if selected_suppliers:
            supplier_data = product_data[product_data['supplier_name'].isin(selected_suppliers)]
            main_metrics = LLMdisplay_supplier_comparison(supplier_data,contract_data)
            display_supplier_comparison(supplier_data, '#B835E8',contract_data)
            # st.title("Recommendations")
            # Extract the first value of Category, Subcategory, and Product Name
            category = main_metrics['Category'].iloc[0]
            subcategory = main_metrics['Subcategory'].iloc[0]
            product_name = main_metrics['Product'].iloc[0]
            
        
            
            st.divider()
            st.header("Recommendations")

            st.markdown(f"""
                <div style="display: flex; gap: 10px; position: absolute; top: -3.3rem; left: 18.5rem;">
                    <div style="background-color: #e0e0e0; padding: 5px 10px; border-radius: 25px;"><b>Category:</b> {category}</div>
                    <div style="background-color: #e0e0e0; padding: 5px 10px; border-radius: 25px;"><b>Subcategory:</b> {subcategory}</div>
                    <div style="background-color: #e0e0e0; padding: 5px 10px; border-radius: 25px;"><b>Product Name:</b> {product_name}</div>
                </div>
            """, unsafe_allow_html=True)

            df = process_supplier_data(main_metrics, tail_metrics)

            # Analyze and display suppliers
            consol_sup, elim_sup = analyze_and_display_suppliers(df, threshold=0.35)

            # st.write(consol_sup)
            # st.write(elim_sup)
            # Display negotiation recommendations
            display_negotiation_recommendations(consol_sup)

            
            # negotiate_payment_terms(consol_sup) 
            # process_and_build_output(consol_sup, elim_sup)
            
            try :# Check if any supplier name in 'consol_sup' is None
                if consol_sup['Supplier Name'].isna().any():
                    st.write("NO Suppliers for Consolidation. Please Adjust the Ranks!")
                else:
                    # Proceed with negotiation if there are valid suppliers
                    negotiate_payment_terms(consol_sup) 
                    process_and_build_output(consol_sup, elim_sup)
            except Exception:
                st.warning("NO Suppliers Matches the criteria for Consolidation. Please Adjust the Ranks!")
            
        else:
            st.info("Please select main suppliers to compare.")
    else:
        st.info("In this case the whole product falls under the tail spend thats why we can't provide a comprehensive recommendations!")

        

def compute_composite_score(supplier_data, delivery_weight=0.5, quality_weight=0.5):
    """
    Function to compute a composite score based on delivery performance and quality checks.
    
    Parameters:
    supplier_data (pd.DataFrame): DataFrame containing delivery_date, actual_receipt_date, and quality_check
    delivery_weight (float): Weight for delivery score (default: 0.5)
    quality_weight (float): Weight for quality score (default: 0.5)
    
    Returns:
    pd.DataFrame: Original dataframe with additional columns for scores
    """
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Calculate delivery delay in days
    supplier_data['delivery_date'] = pd.to_datetime(supplier_data['delivery_date'])
    supplier_data['actual_receipt_date'] = pd.to_datetime(supplier_data['actual_receipt_date'])
    supplier_data['delivery_delay'] = (supplier_data['actual_receipt_date'] - supplier_data['delivery_date']).dt.total_seconds() / (24 * 60 * 60)  # Convert to days
    
    # Normalize delivery delay (negative values are better - meaning early delivery)
    supplier_data['delivery_score'] = scaler.fit_transform(supplier_data[['delivery_delay']]) * -1 + 1  # Invert so higher score means better performance
    
    # Convert boolean quality_check to numeric score (True = 1, False = 0)
    supplier_data['quality_score'] = supplier_data['quality_check'].astype(int)
    
    # Compute the composite score
    supplier_data['composite_score'] = (
        (supplier_data['delivery_score'] * delivery_weight) +
        (supplier_data['quality_score'] * quality_weight)
    )
    
    # Scale to 0-10 range and round
    supplier_data['composite_score'] = (supplier_data['composite_score'] * 10).round(0)
    
    return supplier_data




# Global counter for generating unique keys
widget_counter = 0

def get_unique_key():
    global widget_counter
    widget_counter += 1
    return f'widget_{widget_counter}'

def format_supplier_score(score):
    return f"{round(score)}/10"

# Function to format Financial Score
def format_financial_score(score):
    return f"{round(score)}/5"

def display_supplier_comparison(supplier_data, color, contract_data):
    # Compute composite score
    supplier_data = compute_composite_score(supplier_data)
    
    # Group by supplier and calculate metrics
    comparison_metrics = supplier_data.groupby('supplier_name').agg(
        Category=('parent_category', 'first'),
        Subcategory=('category', 'first'),
        Product=('product_name', 'first'),
        Average_Unit_Price=('unit_price', 'mean'),
        Total_Spend=('po_amount', 'sum'),
        Total_Quantity=('quantity', 'sum'),
        No_of_Transactions=('po_number', 'size'),
        Supplier_Score=('composite_score', 'mean'),
        # Removed financial_score from here
        Payment_Terms=('payment_terms', 'first')
    ).reset_index()
    
    # Add contract details from contract_data
    if not contract_data.empty:
        contract_details = contract_data.groupby('supplier_name').agg(
            Contract_ID=('contract_id', 'first'),
            Contract_Value=('contract_value', 'first')
        ).reset_index()
        
        comparison_metrics = pd.merge(
            comparison_metrics, 
            contract_details,
            on='supplier_name', 
            how='left'
        )
        
        # Add Contract Status
        comparison_metrics['Contract'] = comparison_metrics['Contract_ID'].apply(
            lambda x: 'Yes' if pd.notna(x) and x not in ['', 'nan'] else 'No'
        )
    else:
        # If no contract data, add null columns
        comparison_metrics['Contract_ID'] = None
        comparison_metrics['Contract_Value'] = None
        comparison_metrics['Contract'] = 'No'
    
    # Rename columns
    comparison_metrics = comparison_metrics.rename(columns={
        'supplier_name': 'Supplier Name',
        'Category': 'Category',
        'Subcategory': 'Subcategory',
        'Product': 'Product',
        'Average_Unit_Price': 'Average Unit Price($)',
        'Total_Spend': 'Total Spend($)',
        'Total_Quantity': 'Total Quantity',
        'No_of_Transactions': 'Transactions Count',
        'Supplier_Score': 'Supplier Performance Score',
        # Removed financial_score from here
        'Contract_ID': 'Contract ID',
        'Contract_Value': 'Contract Total Value',
        'Contract': 'Contract Status',
        'Payment_Terms': 'Payment Terms'
    })
    
    plotting_metrics = comparison_metrics.copy()

    comparison_metrics['Supplier Performance Score'] = comparison_metrics['Supplier Performance Score'].apply(format_supplier_score)
    # Removed financial score formatting line
    
    # Display comparison metrics table
    st.write("Supplier Comparison Metrics:")
    st.dataframe(comparison_metrics, hide_index=True)
    
    # Rest of the visualization code remains the same
    show_graphs = st.checkbox('Show Supplier Comparison Graphs', key=get_unique_key())

    if show_graphs:
        # Create the figure with adjusted size and layout
        fig, axs = plt.subplots(2, 2, figsize=(20, 24))

        # Flatten axs for easier iteration
        axs = axs.flatten()

        # List of metrics to plot
        metrics = [
            ('Total Spend($)', 'Total Spend by Supplier'),
            ('Transactions Count', 'Number of Transactions by Supplier'),
            ('Supplier Performance Score', 'Supplier Performance Score'),
            ('Average Unit Price($)', 'Average Unit Price by Supplier')
        ]

        for i, (metric, title) in enumerate(metrics):
            # Ensure the metric is in comparison_metrics
            if metric not in plotting_metrics.columns:
                st.error(f"Metric '{metric}' not found in DataFrame columns.")
                continue

            # Sort data
            sorted_data = plotting_metrics.sort_values(metric, ascending=False).head(10)
            
            # Ensure the column is numeric
            sorted_data[metric] = pd.to_numeric(sorted_data[metric], errors='coerce')

            # Create vertical bar plot with consistent color
            sns.barplot(x='Supplier Name', y=metric, data=sorted_data, ax=axs[i], color=color)

            # Set title and labels with larger font sizes
            axs[i].set_title(title, fontsize=20)
            axs[i].set_xlabel('Supplier Name', fontsize=16)
            axs[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=16)

            # Rotate x-axis labels for better readability
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')

            # Adjust tick label size
            axs[i].tick_params(axis='both', which='major', labelsize=14)

            # Add value labels on top of each bar with larger font size
            for j, v in enumerate(sorted_data[metric]):
                axs[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 2, 0.95])
        plt.subplots_adjust(hspace=0.4, wspace=0.4) 

        # Display graphs in Streamlit
        st.pyplot(fig)

    
def LLMdisplay_supplier_comparison(supplier_data, contract_data):
   """
   Function to display supplier comparison based on spend, price, and other metrics.
   This version returns only the comparison dataframe.
   """
   # Compute composite score
   supplier_data = compute_composite_score(supplier_data)

   def check_contract(x):
       # Convert all values to strings and remove any whitespace
       cleaned = x.astype(str).str.strip()
       # Check if there are any values that are not 'nan' or empty string
       return 'Yes' if any(v not in ['nan', ''] for v in cleaned) else 'No'

   # Group by supplier and calculate key metrics for comparison
   comparison_metrics = supplier_data.groupby('supplier_name').agg(
       Category=('parent_category', 'first'),
       Subcategory=('category', 'first'),
       Product=('product_name', 'first'),
       Average_Unit_Price=('unit_price', 'mean'),
       Total_Spend=('po_amount', 'sum'),
       Total_Quantity=('quantity', 'sum'),
       No_of_Transactions=('po_number', 'size'),
       Supplier_Score=('composite_score', 'mean'),
       financial_score=('financial_score', 'mean'),
       Payment_Terms=('payment_terms', 'first')
   ).reset_index()

   # Add contract details if contract_data is provided
   if contract_data is not None and not contract_data.empty:
       contract_details = contract_data.groupby('supplier_name').agg(
           Contract_ID=('contract_id', 'first'),
           Contract_Total_Value=('contract_value', 'first')
           
       ).reset_index()
       
       comparison_metrics = pd.merge(
           comparison_metrics, 
           contract_details, 
           on='supplier_name', 
           how='left'
       )
       
       # Add Contract Status
       comparison_metrics['Contract'] = comparison_metrics['Contract_ID'].apply(
           lambda x: 'Yes' if pd.notna(x) and x not in ['', 'nan'] else 'No'
       )
   else:
       # Add null columns for contract details if no contract data
       comparison_metrics['Contract_ID'] = None
       comparison_metrics['Contract_Total_Value'] = None
       
       comparison_metrics['Contract'] = 'No'

   # Rename the columns
   comparison_metrics = comparison_metrics.rename(columns={
       'supplier_name': 'Supplier Name',
       'Category': 'Category',
       'Subcategory': 'Subcategory',
       'Product': 'Product',
       'Average_Unit_Price': 'Average Unit Price($)',
       'Total_Spend': 'Total Spend($)',
       'Total_Quantity': 'Total Quantity',
       'No_of_Transactions': 'Transactions Count',
       'Supplier_Score': 'Supplier Performance Score',
       'financial_score': 'Financial Score',
       'Contract_ID': 'Contract ID',
       'Contract_Total_Value': 'Contract Total Value',
       'Payment_Terms': 'Payment Terms',
       'Contract': 'Contract Status'
   })

   # Return the comparison metrics DataFrame
   return comparison_metrics

# List of functions to be imported when module is imported
__all__ = [
    'supplier_performance_analysis',
    'compute_composite_score',
    'display_supplier_comparison'
]

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(e)
