import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

du_lieu = pd.read_csv("facebook_comments.csv")
print("Xem trước dữ liệu:")
print(du_lieu.head())

def lam_sach_van_ban(van_ban):
    van_ban = str(van_ban).lower()
    van_ban = re.sub(r"[^a-zA-Zà-ỹÀ-Ỹ\s]", "", van_ban)
    return van_ban

du_lieu['van_ban_sach'] = du_lieu['comment'].apply(lam_sach_van_ban)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(du_lieu['van_ban_sach'])
y = du_lieu['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mo_hinh = LogisticRegression(solver='lbfgs', max_iter=1000)
mo_hinh.fit(X_train, y_train)

du_doan = mo_hinh.predict(X_test)
print("Báo cáo đánh giá mô hình (0 = Tiêu cực, 1 = Tích cực, 2 = Trung tính):")
print(classification_report(y_test, du_doan, zero_division=0))

du_doan_toan_bo = mo_hinh.predict(X)

so_tieu_cuc = sum(du_doan_toan_bo == 0)
so_tich_cuc = sum(du_doan_toan_bo == 1)
so_trung_tinh = sum(du_doan_toan_bo == 2)

nhan = ['Tiêu cực', 'Tích cực', 'Trung tính']
so_luong = [so_tieu_cuc, so_tich_cuc, so_trung_tinh]
mau_sac = ['lightcoral', 'lightgreen', 'lightgray']

plt.figure(figsize=(6,6))
plt.pie(so_luong, labels=nhan, colors=mau_sac, autopct='%1.1f%%', startangle=140)
plt.title("Tỷ lệ cảm xúc trong comment Facebook", y=1.08)
plt.axis('equal')
plt.show()
