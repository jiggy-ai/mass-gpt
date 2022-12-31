from botocore.exceptions import ClientError

"""
CUSTOM EXCEPTIONS - DESIGNED AS MORE USEFUL WRAPPERS TO BOTOCORE'S ARCANE BS EXCUSE-FOR-EXCEPTIONS
BASICALLY ENCAPSULATE BOTO EXCEPTIONS W/MORE INFORMATION.
CLIENT CODE UNLIKELY TO KNOW HOW TO CATCH BOTOCORE EXCEPTIONS, BUT THESE ARE EXPOSED THROUGH S3 CLASS SO EZ
"""


class BucketException(Exception):
    """
    PARENT CLASS THAT ENSURES THAT ALL BUCKET ERRORS ARE CONSISTENT
    """

    def __init__(self, message, bucket):
        self.bucket = bucket
        self.message = f'{message}'
        super().__init__(self.message)


class NoSuchKey(BucketException):
    """
    RAISED IF YOU TRY TO ACCESS A NON-EXISTENT OBJECT, SINCE IT HAS MOST LIKELY EXPIRED
    """

    def __init__(self, key, bucket):
        self.key = key
        self.bucket = bucket
        self.message = f'No object in bucket {bucket} matches {key}. Has it expired?'
        super().__init__(self.message, self.bucket)


class NoSuchBucket(BucketException):
    """
    RAISED IF YOU TRY TO ACCESS A NONEXISTENT BUCKET
    """

    def __init__(self, bucket_name):
        self.bucket = bucket_name
        self.message = f'Bucket {bucket_name} does not exist!'
        super().__init__(self.message, self.bucket)


class BucketAccessDenied(BucketException):
    """
    RAISED IF ACCESS TO A BUCKET IS DENIED - LIKELY BECAUSE IT DOESN'T EXIST
    """

    def __init__(self, bucket_name):
        self.bucket = bucket_name
        self.message = f'Unable to access bucket {self.bucket}. Does it exist?'

        super().__init__(self.message, self.bucket)


class UnknownBucketException(BucketException):
    """
    RAISED IF AN UNKNOWN S3 EXCEPTION OCCURS
    """

    def __init__(self, bucket_name, e: ClientError):
        self.bucket = bucket_name
        error_code: str = e.response.get('Error').get('Code')
        error_message: str = e.response.get('Error').get('Message')
        self.message = f'Unknown Bucket Exception {error_code}: {error_message}'
        super().__init__(self.message, self.bucket)
