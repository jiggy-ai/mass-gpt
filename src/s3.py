import s3_bucket as S3
import boto3
import os
from botocore.exceptions import ClientError

BUCKET_NAME = 'jiggy-assets'
ENDPOINT_URL = os.environ.get("JIGGY_STORAGE_ENDPOINT_URL", "https://us-southeast-1.linodeobjects.com")

STORAGE_KEY_ID = os.environ['JIGGY_STORAGE_KEY_ID']
STORAGE_SECRET_KEY = os.environ['JIGGY_STORAGE_KEY_SECRET']


S3.Bucket.prepare(STORAGE_KEY_ID,
                  STORAGE_SECRET_KEY,
                  endpoint_url=ENDPOINT_URL)


bucket = S3.Bucket(BUCKET_NAME)


linode_obj_config = {
    "aws_access_key_id": STORAGE_KEY_ID,
    "aws_secret_access_key": STORAGE_SECRET_KEY,
    "endpoint_url": ENDPOINT_URL}


s3_client = boto3.client('s3', **linode_obj_config)


def create_presigned_url(object_name, expiration=300):
    """
    Generate a presigned URL to share an S3 object
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': BUCKET_NAME,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response

    
