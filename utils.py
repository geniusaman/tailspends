import streamlit as st

def create_kpi_metric(label, value, delta=None, delta_color="normal"):
    if delta:
        st.metric(label, value, delta, delta_color=delta_color)
    else:
        st.metric(label, value)

def create_recommendation_button(key):
    if st.button('Show Recommendations', key=key):
        st.subheader('Recommendations')
        recommendations = [
            "**Supplier Consolidation:** Consider consolidating purchases with top-performing suppliers to leverage better pricing and terms.",
            "**Supplier Performance:** Regularly review supplier performance to ensure quality and compliance with contract terms.",
            "**Tail Spend Management:** Implement strategies to better control tail spend, potentially renegotiating contracts or finding alternative suppliers.",
            "**Non-Contracted Spend:** Investigate instances where items are purchased from non-contracted suppliers. Reinforce purchasing policies and consider renegotiating contracts."
        ]
        for rec in recommendations:
            st.write(rec)