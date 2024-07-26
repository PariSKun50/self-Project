from io import StringIO
import yfinance as yf
import pandas as pd
import boto3
import datetime
import time


def main_handler(event, context):
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date_short = (datetime.datetime.now() - datetime.timedelta(days=10*365)).strftime('%Y-%m-%d')
    start_date_long = (datetime.datetime.now() - datetime.timedelta(days=20*365)).strftime('%Y-%m-%d')

    # Download daily data
    short_data = yf.download(tickers=['MARUTI.NS'], start=start_date_short, end=today, interval="1d")
    short_data_csv = short_data.to_csv()

    # Download weekly data
    long_data = yf.download(tickers=['MARUTI.NS'], start=start_date_long, end=today, interval="1wk")
    long_data_csv = long_data.to_csv()
    
    # Upload to S3
    upload_to_s3(short_data_csv, 's3://fpshorttermbucket/short_data.csv')
    upload_to_s3(long_data_csv, 's3://fplongtermbucket/long_data.csv')

    # Create and deploy SageMaker model
    create_and_deploy_sagemaker_model('fpshortterm_backend', 'fpshorttermbucket','s3://fpshorttermbucket/short_data.csv')
    create_and_deploy_sagemaker_model('fplongterm_backend', 'fplongtermbucket', 's3://fplongtermbucket/long_data.csv')

def upload_to_s3(data, s3_path):
    access_key = 'remove-AKIAVEIXKUD2'
    secret_key = 'remove-nGRzUSYFLuU7QTNJX6u'
    region_name = 'us-east-1'
    
    boto3.setup_default_session(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region_name)
    
    s3_client = boto3.client('s3')
    # Extract bucket name and key from s3_path
    bucket_name, key = s3_path.replace('s3://', '').split('/', 1)


    csv_buffer = StringIO()
    csv_buffer.write(data)
    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=key)

def create_and_deploy_sagemaker_model(endpoint_name, s3_bucket,s3_path):
    
    # Generate a unique model name based on the current date
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    model_name = f'{endpoint_name}-model-{today}-19'

    # Set the target column and weightage for all features
    target_column = 'Adj Close'
    feature_weightage = 'auto'

    # Set the training method to Auto
    training_method = 'Auto'

    # Set up SageMaker client
    sagemaker = boto3.client('sagemaker')

    # Create Autopilot experiment
    experiment_name = create_autopilot_experiment(sagemaker, model_name, s3_bucket,s3_path, target_column, feature_weightage, training_method)
    
    # Wait for the best candidate to complete training
    wait_for_training_completion(sagemaker, experiment_name, model_name)
 
    parts = endpoint_name.split('_')
    best_candidate_name = parts[0]

    deploy_model(sagemaker,experiment_name, endpoint_name,best_candidate_name)

def create_autopilot_experiment(sagemaker, experiment_name, s3_bucket,s3_input_path, target_column, feature_weightage, training_method):
    # Specify Autopilot job configuration
    autopilot_job_config = {
        'CompletionCriteria': {
            'MaxRuntimePerTrainingJobInSeconds': 1800,  # Set the maximum runtime
            'MaxCandidates': 250,
            'MaxAutoMLJobRuntimeInSeconds': 7200,
        },
        'SecurityConfig': {
            'EnableInterContainerTrafficEncryption': False
        }
    }

    # Start Autopilot job
    sagemaker.create_auto_ml_job(
        AutoMLJobName=experiment_name,
        InputDataConfig=[
            {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': s3_input_path
                    }
                },
                'TargetAttributeName': target_column
            }
        ],
        OutputDataConfig={
            'S3OutputPath': f's3://{s3_bucket}/output'
        },
        RoleArn='arn:aws:iam::352773120245:role/service-role/AmazonSageMaker-ExecutionRole-20231205T004726',
        AutoMLJobConfig=autopilot_job_config
    )

    return experiment_name
def wait_for_training_completion(sagemaker, experiment_name, model_name, poll_interval_seconds=60):
    while True:
        response = sagemaker.describe_auto_ml_job(AutoMLJobName=experiment_name)

        # Print the response for debugging
        print("API Response:", response)
        print("****************")
        # Check if response is a string (error response) and print the error
        if isinstance(response, str):
            print(f"Error: {response}")
            print("**********1*************")
            break

        auto_ml_job_status = response.get('AutoMLJobStatus', {})
    
        # Check if the job is in progress
        if auto_ml_job_status == 'InProgress':
            print("Model training is still in progress. Waiting...")
            time.sleep(poll_interval_seconds)
            continue
        time.sleep(poll_interval_seconds)


def deploy_model(sagemaker,endpoint_name, experiment_name, best_candidate_name):

    listmodel=sagemaker.list_models()
    models = listmodel['Models']

    # Filter models starting with 'fpshortterm' and 'fplongterm'
    fpterm_models = [model for model in models if model['ModelName'].startswith(best_candidate_name)]
    # Sort models by creation time (latest first)
    fpterm_models.sort(key=lambda x: x['CreationTime'], reverse=True)
    # Get the latest models
    latest_fpterm_model = fpterm_models[0] if fpterm_models else None
    latest_fpterm_name = latest_fpterm_model['ModelName']
    print(latest_fpterm_name)
    # Specify model deployment configuration
    model_deploy_config = {
        'EndpointName': endpoint_name,
        'EndpointConfigName': experiment_name,  # Use the Autopilot experiment name as the endpoint config
    }
    print(model_deploy_config)
    production_variant = {
        'InstanceType': 'ml.t2.medium',
        'InitialInstanceCount': 1,
        'ModelName': latest_fpterm_name,  # Use the best candidate name as the model name
        'VariantName': 'AllTraffic'
    }
    print(production_variant)
    create_endpoint_config_response = sagemaker.create_endpoint_config(
        EndpointConfigName=experiment_name,
        ProductionVariants=[production_variant]
    )
    # Deploy the model
    sagemaker.create_endpoint(**model_deploy_config)

# Uncomment and run the script for local testing
# main_handler(None, None)



