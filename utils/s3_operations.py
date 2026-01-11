import sys
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from exception.custom_exception import ProjectCustomException
import logger
from utils.APIKey_loader import APIKeyManager

class S3ReadUpload:
    def __init__(self):
        api_key_mgr = APIKeyManager(['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION'])
        self.aws_access_key_id = api_key_mgr.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = api_key_mgr.get('AWS_SECRET_ACCESS_KEY')
        self.region_name = api_key_mgr.get('AWS_REGION')
        if not self.aws_access_key_id or not self.aws_secret_access_key or not self.region_name:
            raise ValueError("AWS credentials and region must be provided in the env file.")

    def upload_file_to_s3(self, file_name, bucket_name, object_name):
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

        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
            else:
                s3_client = boto3.client('s3')

            # Upload the file
            s3_client.upload_file(file_name, bucket_name, object_name)
            print(f"File {file_name} uploaded to {bucket_name}/{object_name}")
            return True
        except Exception as e:
            raise ProjectCustomException(f"Failed to upload {file_name} to S3", sys)


    def read_file_from_s3(self, bucket_name, object_name):
        """
        Read a file from an S3 bucket.

        :param bucket_name: S3 bucket name
        :param object_name: S3 object name
        :param aws_access_key: AWS access key ID (optional)
        :param aws_secret_key: AWS secret access key (optional)
        :param region_name: AWS region name (optional)
        :return: File content as bytes, or None if error occurs
        """
        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
            else:
                s3_client = boto3.client('s3')

            response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
            file_content = response['Body'].read()
            return file_content
        except Exception as e:
            raise ProjectCustomException(f"Failed to read {object_name} from S3", sys)
    

if __name__ == "__main__":
    # Load Configuration:
    from config_loader import load_config
    config = load_config("config.yaml")
    s3_config = config.get('AWS-S3', {})
    bucket_name = s3_config.get('bucket_name')
    object_prefix = s3_config.get('preindex_folder_name')
    users = config.get('user_names', {})
    usse = users.get('user1', 'default_user')
 
    # Example usage
    file_name = "./data/Rudy-2025.pdf"
    object_name = f"{object_prefix}/{usse}/Rudy-2025.pdf"

    s3_ops = S3ReadUpload()
    s3_ops.upload_file_to_s3(file_name, bucket_name, object_name)

    file_content = s3_ops.read_file_from_s3(bucket_name, object_name)
    print(f"Read {len(file_content)} bytes from S3 object {object_name}")