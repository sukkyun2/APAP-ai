import boto3
import google.generativeai as genai

s3 = boto3.client('s3')

GOOGLE_API_KEY = "AIzaSyCLesrwqsh0WDxRDiSm6GSocDYfB9OzcUw"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')


def get_listall_from_aws(bucket_name):
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name)
    temp_list = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                file_url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"
                temp_list.append(f"File: {file_key}, URL: {file_url}")

    return temp_list


def call_gemini(img, pram_list):
    temp_list = []
    class_count = {}

    for detection in pram_list:
        class_name = detection.class_name
        if class_name in class_count:
            class_count[class_name] += 1
        else:
            class_count[class_name] = 1

        new_class_name = f"{class_name}{class_count[class_name]}"
        temp_list.append(new_class_name)

    response = model.generate_content(
        (img, "이 이미지에 대하여, 다음 object list의 요소 간 관계를 중심으로 어떤 안전사고가 예상되는 상황인지 판단해." + str(temp_list)))

    return response.text
