import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def upload_file_to_s3(file_name, bucket_name, object_name=None, aws_access_key=None, aws_secret_key=None, region_name=None):
    """
    Upload a file to an S3 bucket.

    :param file_name: File to upload
    :param bucket_name: S3 bucket name
    :param object_name: S3 object name. If not specified, file_name is used
    :param aws_access_key: AWS access key ID (optional)
    :param aws_secret_key: AWS secret access key (optional)
    :param region_name: AWS region name (optional)
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Initialize S3 client
    try:
        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )
        else:
            s3_client = boto3.client('s3')

        # Upload the file
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File {file_name} uploaded to {bucket_name}/{object_name}")
        return True
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False

if __name__ == "__main__":
    # Load Configuration:
    from config_loader import load_config
    config = load_config("config.yaml")
    s3_config = config.get('AWS-S3', {})
    bucket_name = s3_config.get('bucket_name')
    object_prefix = s3_config.get('preindex_folder_name')

    # Load Environment Variables
    import os
    from dotenv import load_dotenv
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    # Example usage
    file_name = "./data/text.txt"
    file_name = "/Users/arindam/Machine Learning/GenAI_2025/AgenticAI/13_Personal_ChatGPT/data/Rudy-2025.pdf"
    object_name = f"{object_prefix}/Rudy-2025.pdf"

    upload_file_to_s3(file_name, bucket_name, object_name, aws_access_key, aws_secret_key, aws_region)