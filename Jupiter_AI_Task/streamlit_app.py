import streamlit as st
import requests

API_URL = 'http://localhost:5000/'

def get_response(query):
    data = {
        'query': query
    }

    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return 'Oops! Something went wrong.'

def main():
    st.title('Jupiter AI Query')
    query = st.text_input('Enter your query')

    if st.button('Submit'):
        if query:
            response = get_response(query)
            st.text_area('Response', response)
        else:
            st.warning('Please enter a query.')

if __name__ == '__main__':
    main()

