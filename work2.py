import numpy as np
import matplotlib.pyplot as plt

# 1. กำหนดค่าพารามิเตอร์สำหรับการแจกแจงปกติชุดที่ 1 (สีแดง 'a')
mean1 = [-3, 5]
# ลดความแปรปรวน (Variance) ลงเหลือ 0.5 (จากเดิม 1) เพื่อให้จุดกระจุกตัวแน่นขึ้น
cov1 = np.array([[1, 0], [0, 1]]) 

# 2. กำหนดค่าพารามิเตอร์สำหรับการแจกแจงปกติชุดที่ 2 (สีน้ำเงิน 'b')
mean2 = [3, 5]
# ลดความแปรปรวน (Variance) ลงเหลือ 0.5
cov2 = np.array([[1, 0], [0, 1]]) 

# 3. สร้างจุดข้อมูล (Sampling)
# สุ่มจุดข้อมูล 500 จุด
pts1 = np.random.multivariate_normal(mean1, cov1, size=500)
pts2 = np.random.multivariate_normal(mean2, cov2, size=500)

# 4. พล็อตกราฟ (Plotting)
plt.figure(figsize=(8, 6))
plt.scatter(pts1[:, 0], pts1[:, 1], marker='.', s=20, alpha=0.6, color='red', label='a')
plt.scatter(pts2[:, 0], pts2[:, 1], marker='.', s=20, alpha=0.6, color='blue', label='b')

# เพิ่มเส้นแบ่งตรงกลางที่ X = 0 (Decision Boundary)
plt.axvline(x=0, color='black', linestyle='-', linewidth=2) # ลบ label ออกเพื่อไม่ให้ซ้ำกับ Legend

# 5. การตั้งค่าแกนและรูปแบบการแสดงผล (Axis Configuration)
plt.axis('equal')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.xlim(-7.5, 7.5)
plt.ylim(0, 10)

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Multivariate Normal Distribution Plot (Reduced Overlap)', fontsize=14)
plt.show()