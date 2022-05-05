import json
import urllib.parse
import boto3
import email
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences
print('Loading function')
s3 = boto3.client('s3')
region = 'us-east-1'

def sendSESMail(message,email):
    ses_client = boto3.client('ses', region_name=region)
    response = ses_client.send_email(
        Source='st4324@nyu.edu',
        Destination={
            'ToAddresses': [email]
        },
        ReplyToAddresses=['st4324@nyu.edu'],
        Message={
            'Subject': {
                'Data': 'NYU Email Bot',
                'Charset': 'utf-8'
            },
            'Body': {
                'Text': {
                    'Data': message,
                    'Charset': 'utf-8'
                },
                'Html': {
                    'Data': message,
                    'Charset': 'utf-8'
                }
            }
        }
    )


def lambda_handler(event, context):
    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read().decode('utf-8')

        # parse body to obtain To: From: Date: and 
        fromEmail = ''
        toEmail = ''
        date = ''
        subject = ''
        emailBody = '' # limit to 240 characters
        
        emailContents = email.message_from_string(body)
        
        if emailContents.is_multipart():
            for payload in emailContents.get_payload():
                if payload.get_content_type() == 'text/plain':
                    emailBody = payload.get_payload()
        else:
            emailBody = emailContents.get_payload()
        emailBody = emailBody.replace("\r", " ").replace("\n", " ")
            
        fromEmail = emailContents.get('From')
        fromEmail = fromEmail[fromEmail.find('<') + 1:-1]
        toEmail = emailContents.get('To')
        date = emailContents.get('Date')
        subject = emailContents.get('Subject')

        # call Sagemaker to obtain results
        ENDPOINT_NAME = os.environ['PRED_URL']
        vocabulary_length=9013
        emailBody_for_processing=[emailBody]
        runtime= boto3.client('runtime.sagemaker') 
        one_hot_test_messages = one_hot_encode(emailBody_for_processing, vocabulary_length)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
        payload = json.dumps(encoded_test_messages.tolist())
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=payload)
        response_body = response['Body'].read().decode('utf-8')
        result = json.loads(response_body)
        classification = ''
        if(result["predicted_label"][0][0]==1):
             classification = ''+'SPAM'
             classificationScore = result['predicted_probability'][0][0]*100
        else:
             classification = ''+'HAM'
             classificationScore = result['predicted_probability'][0][0]*100

        
        # send response through SES
        if len(emailBody) > 240:
            emailBody = emailBody[:240]
        line1 = 'We received your email sent at ' + date + ' with the subject ' + subject + '.\n'
        line2 = 'Here is a 240 character sample of the email body: \n'
        line3 = emailBody + '\n'
        line4 = 'The email was categorized as ' + classification + ' with a ' + str(classificationScore) + '% confidence.'
        print(line1 + line2 + line3 + line4)
        sendSESMail(line1 + line2 + line3 + line4,fromEmail)
        return "success"
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
