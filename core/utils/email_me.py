import smtplib
from email.message import EmailMessage
import traceback
import argparse
def send_email(subject, body):
    # Email configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465  # Use 465 for SSL
    sender_email = 'anonymous@gmail.com'
    # Use your generated App Password here
    sender_password = 'pfrk ieas nkzt helz' 
    recipient_emails=[]
    recipient_emails.append('anonymous@gmail.com')
    

    for recipient_email in recipient_emails:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email
        try:
            # Connect to the SMTP server and send the email
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
                smtp.login(sender_email, sender_password)
                smtp.send_message(msg)
            print('Email notification sent successfully!')
        except Exception as e:
            print(f'Error sending email notification: {e}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="Test Notification", help="Test Notification Subject")
    parser.add_argument("--body", type=str, default="Hi,\nThe default body message for the test. If you get this, either somebody ran my script for fun or its a test message. Hopefully I am not in spam \nBest,\nNikhil's Server Script\n")
    args = parser.parse_args()

    send_email(args.subject,args.body)
     
 