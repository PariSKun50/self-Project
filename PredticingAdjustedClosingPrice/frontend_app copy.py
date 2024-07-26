import streamlit as st
import pandas as pd
import boto3
import sagemaker

print(boto3.__version__)
print(sagemaker.__version__)

#access_key = '*******'
#secret_key = '*****'
region_name = 'us-east-1'

session = boto3.Session(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region_name
)

# Now you can use the session to create clients for various AWS services, including SageMaker.


# Set Pandas display options to hide row numbers
pd.set_option('display.max_rows', None)

# Streamlit App Title
st.title("Foundation Project 2")
st.write("Stocks Of Maruti Suzuki India Limited")
st.write("Please select Investment Type as Short Term if you want next day prediction and Long Term if you want a Week prediction")

time_frame = st.selectbox("Select Investment Time Frame", ["Short Term", "Long Term"])

if time_frame == "Short Term":
    df_short = pd.read_csv('short_data.csv')

    # Display the last 5 rows without row numbers
    st.markdown("### Last 5 Rows of Short-Term Data:")
    st.markdown(df_short.tail().to_markdown(index=False))

    # Allow the user to select the columns for X and Y axes
    x_column_short = st.selectbox("Date", df_short.columns)
    y_column_short = st.selectbox("Adj Close", df_short.columns, index=df_short.columns.get_loc("Adj Close"))

    # Display the line chart
    st.line_chart(df_short.set_index(x_column_short)[y_column_short])

    ## Extract the last value of "Adj Close"
    last_adj_close = df_short["Adj Close"].iloc[-1]

    ep_name = 'fpshortterm-model-2023-12-06-19'
    last_row = df_short.iloc[-1]

    # Extract values from the last row
    date = pd.to_datetime(last_row['Date']) + pd.DateOffset(1)
    open_value = last_row['Open']
    high_value = last_row['High']
    low_value = last_row['Low']
    close_value = last_row['Close']
    volume_value = last_row['Volume']
    #sample = f'{date.strftime("%Y-%m-%d")},{open_value:.2f},{high_value:.2f},{low_value:.2f},{close_value:.2f},{int(volume_value)}'
    sample = '2023-12-08,10730.00,10748.00,10551.00,10618.55,548373'
    st.write("Predicted Stocks Of Maruti Suzuki India Limited for Date: 2023-12-08")
    # Call URL and display the response
    sm_st = session.client('runtime.sagemaker')
    response = sm_st.invoke_endpoint(EndpointName = ep_name, ContentType = 'text/csv', Accept = 'text/csv', Body = sample)
    response = response['Body'].read().decode("utf-8")
    print(response)
    result_from_url = float(response)  # Assuming the result is a numeric value

    # Compare and display result
    st.write("Last Adj Close Value: INR", last_adj_close)
    st.write("Predicted Adj Close for Short Term: INR", result_from_url)

    if result_from_url > last_adj_close:
        st.write("Result: Positive")
    else:
        st.write("Result: Negative")

else:
    df_long = pd.read_csv('long_data.csv')

    # Display the last 5 rows without row numbers
    st.markdown("### Last 5 Rows of Long-Term Data:")
    st.markdown(df_long.tail().to_markdown(index=False))

    # Allow the user to select the columns for X and Y axes
    x_column_long = st.selectbox("Date", df_long.columns)
    y_column_long = st.selectbox("Adj Close", df_long.columns, index=df_long.columns.get_loc("Adj Close"))

    # Display the line chart
    st.line_chart(df_long.set_index(x_column_long)[y_column_long])

    # Extract the last value of "Adj Close"
    last_adj_close = df_long["Adj Close"].iloc[-1]

    ep_name = 'fplongterm-model-2023-12-05'
    last_row = df_long.iloc[-1]

    # Extract values from the last row
    date = pd.to_datetime(last_row['Date']) + pd.DateOffset(weeks=1)
    open_value = last_row['Open']
    high_value = last_row['High']
    low_value = last_row['Low']
    close_value = last_row['Close']
    volume_value = last_row['Volume']
    sample = f'{date.strftime("%Y-%m-%d")},{open_value:.2f},{high_value:.2f},{low_value:.2f},{close_value:.2f},{int(volume_value)}'

    sm_lt = session.client('runtime.sagemaker')
    response = sm_lt.invoke_endpoint(EndpointName = ep_name, ContentType = 'text/csv', Accept = 'text/csv', Body = sample)
    response = response['Body'].read().decode("utf-8")
    print(response)
    result_from_url = float(response)  # Assuming the result is a numeric value

    # Compare and display result
    st.write("Last Adj Close Value: INR", last_adj_close)
    st.write("Predicted Adj Close for Long Term: INR", result_from_url)

    if result_from_url > last_adj_close:
        st.write("Result: Positive")
    else:
        st.write("Result: Negative")
