#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit Cloud Dashboard Placeholder

This file serves as a placeholder and basic example for a Streamlit-based cloud dashboard.
It would connect to the remote database (e.g., MongoDB) to fetch and display session data.
"""

import streamlit as st
# import pandas as pd
# import plotly.express as px
# from ...database.mongodb_client import MongoDBClient # Adjust path as needed
# from ...config import load_config # To get DB config

# --- Configuration (Example) ---
# This would typically be loaded from a config file or environment variables
# APP_CONFIG = load_config() # Assuming a global config loader
# DB_CONFIG = APP_CONFIG.get('database', {}).get('mongodb', {})
# MONGODB_CONNECTION_STRING = DB_CONFIG.get('connection_string', "mongodb://localhost:27017/")
# MONGODB_DATABASE_NAME = DB_CONFIG.get('database_name', "nexaris_cle_data")

# @st.cache_resource # Cache the DB connection
# def get_db_client():
#     print("Initializing MongoDB client for Streamlit...")
#     client = MongoDBClient(MONGODB_CONNECTION_STRING, MONGODB_DATABASE_NAME)
#     if client.connect():
#         return client
#     else:
#         st.error("Failed to connect to the database. Please check the connection string and ensure MongoDB is running.")
#         return None

# @st.cache_data(ttl=600) # Cache data for 10 minutes
# def load_session_data(_db_client, query_filter=None, limit=100):
#     if _db_client:
#         st.info(f"Fetching session data... (Limit: {limit})")
#         # Example: Fetching all sessions, limited for performance
#         # Add more sophisticated querying based on user input
#         sessions = _db_client.get_all_sessions(collection_name="sessions", query_filter=query_filter or {})
#         if sessions:
#             df = pd.DataFrame(sessions)
#             # Basic preprocessing: convert ObjectId, handle missing data, etc.
#             if '_id' in df.columns:
#                 df['_id'] = df['_id'].astype(str)
#             if 'timestamp_start' in df.columns:
#                 df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], errors='coerce')
#             return df.head(limit) # Return limited for display
#         else:
#             st.warning("No session data found.")
#             return pd.DataFrame()
#     return pd.DataFrame()

def main_dashboard():
    st.set_page_config(layout="wide", page_title="NEXARIS CLE - Cloud Dashboard")
    st.title("🧠 NEXARIS CLE - Cloud Dashboard (Placeholder)")
    st.markdown("--- Placeholder for future cloud dashboard using Streamlit ---")

    # db_client = get_db_client()

    # if not db_client:
    #     st.stop()

    st.sidebar.header("Filters & Options")
    # user_id_filter = st.sidebar.text_input("Filter by User ID (optional)")
    # task_name_filter = st.sidebar.text_input("Filter by Task Name (optional)")
    # date_range_filter = st.sidebar.date_input("Filter by Date Range (optional)", value=[], min_value=None, max_value=None)

    # query = {}
    # if user_id_filter:
    #     query['user_id'] = user_id_filter
    # if task_name_filter:
    #     query['task_name'] = {'$regex': task_name_filter, '$options': 'i'} # Case-insensitive search
    # if len(date_range_filter) == 2:
    #     query['timestamp_start'] = {'$gte': pd.to_datetime(date_range_filter[0]), '$lte': pd.to_datetime(date_range_filter[1])}
    
    # sessions_df = load_session_data(db_client, query_filter=query)

    st.header("Recent Sessions")
    # if not sessions_df.empty:
    #     st.dataframe(sessions_df[['_id', 'user_id', 'task_name', 'timestamp_start', 'duration_seconds']].head(10))
        
    #     st.subheader("Select Session for Details")
    #     selected_session_id = st.selectbox("Session ID", options=sessions_df['_id'].tolist())
        
    #     if selected_session_id:
    #         selected_session_data = sessions_df[sessions_df['_id'] == selected_session_id].iloc[0].to_dict()
    #         st.json(selected_session_data, expanded=False)

    #         # Example: Plot cognitive load scores if available
    #         if 'cognitive_load_scores' in selected_session_data and isinstance(selected_session_data['cognitive_load_scores'], list):
    #             st.subheader("Cognitive Load Trend")
    #             # fig = px.line(y=selected_session_data['cognitive_load_scores'], labels={'y':'Cognitive Load', 'x':'Time Point'}, title="Cognitive Load Over Time")
    #             # st.plotly_chart(fig, use_container_width=True)
    #             st.line_chart(selected_session_data['cognitive_load_scores'])
    # else:
    #     st.info("No session data to display based on current filters.")

    st.info("This is a placeholder Streamlit application. Full functionality to be implemented.")
    st.warning("Database connection and data fetching are currently commented out.")

if __name__ == '__main__':
    # To run this app: streamlit run streamlit_app_placeholder.py
    main_dashboard()