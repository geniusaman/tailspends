import streamlit as st
from pymongo import MongoClient
import pandas as pd

def init_connection():
    return MongoClient("mongodb+srv://aryan:3oaCNFmGaMicaLSK@cluster0.2xhwlko.mongodb.net/?retryWrites=true&w=majority&tls=true&appName=Cluster0")

def load_data():
    client = init_connection()
    db = client.temp_database
    items = db.stainless_po_data.find()
    return pd.DataFrame(list(items))


def load_contract_data():
    client = init_connection()
    db = client.temp_database
    items = db.Contracts.find()
    return pd.DataFrame(list(items))
