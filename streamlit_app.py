import streamlit as st
import requests
from bs4 import BeautifulSoup

# Create a text input widget to accept a URL
url = st.text_input('Enter a URL')

# Create a button to fetch the content
if st.button('Fetch Content'):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the text from the HTML content
    text = soup.get_text()
    
    # Display the text
    st.write(text)
