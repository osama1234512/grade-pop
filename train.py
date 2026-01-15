from pycaret.classification import *
import pandas as pd

# 1. تحميل البيانات (تأكد من وجود المسافة والشرطة في اسم الملف)
data = pd.read_csv('Student_performance_data _.csv')

# 2. تنظيف أسماء الأعمدة من أي فراغات زائدة
data.columns = [col.strip() for col in data.columns]

# 3. إعداد البيئة (تم تصحيح الأسماء لتطابق الملف 100%)
clf = setup(
    data, 
    target = 'GradeClass',  # الحرف G و C كبيران
    session_id = 123,
    numeric_features = ['Age', 'StudyTimeWeekly', 'Absences'],
    categorical_features = [
        'Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
        'ParentalSupport', 'Extracurricular', 'Sports', # تم تصحيح Extracurricular
        'Music', 'Volunteering'
    ],
    ignore_features = ['StudentID', 'GPA'] # تجاهل GPA لأنه يغشش الموديل النتيجة
)

# 4. مقارنة الموديلات
print("جاري البحث عن أفضل موديل ذكاء اصطناعي...")
best_model = compare_models()

# 5. حفظ الموديل
save_model(best_model, 'best_student_performance_model')

create_api(best_model, 'student_performance_api')