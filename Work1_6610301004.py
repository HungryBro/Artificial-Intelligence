# Pattern Recognition - Probability Theory
# Two boxes: Red and Blue
# Fruits: Apple (a) and Orange (o)

# -----------------------------
# 1) PRIOR PROBABILITIES
# -----------------------------
# โอกาสเลือกกล่องแต่ละสี
p_red_box = 0.40      # กล่องแดง
p_blue_box = 0.60     # กล่องน้ำเงิน

# เพื่อให้สอดคล้องกับการพิมพ์ผลลัพธ์เดิม
p_red = p_red_box
p_blue = p_blue_box

# -----------------------------
# 2) NUMBER OF FRUITS IN EACH BOX
# -----------------------------
# red box: 2 apples, 4 oranges
red_apple_count = 2
red_orange_count = 4

# blue box: 3 apples, 1 orange
blue_apple_count = 3
blue_orange_count = 1

# Total fruits in each box
n_red = red_apple_count + red_orange_count
n_blue = blue_apple_count + blue_orange_count

# -----------------------------
# 3) LIKELIHOOD: P(F | Box)
# -----------------------------
# P(F = apple | red box)
p_a_given_red = red_apple_count / n_red
# P(F = orange | red box)
p_o_given_red = red_orange_count / n_red

# P(F = apple | blue box)
p_a_given_blue = blue_apple_count / n_blue
# P(F = orange | blue box)
p_o_given_blue = blue_orange_count / n_blue

# -----------------------------
# 4) MARGINAL PROBABILITY: P(F)
# -----------------------------
# P(F = a) = P(a|red)*P(red) + P(a|blue)*P(blue)
p_a = p_a_given_red * p_red_box + p_a_given_blue * p_blue_box

# P(F = o) = P(o|red)*P(red) + P(o|blue)*P(blue)
p_o = p_o_given_red * p_red_box + p_o_given_blue * p_blue_box

# -----------------------------
# 5) POSTERIOR: P(red | orange)
# -----------------------------
# P(red | o) = P(o|red) * P(red) / P(o)
p_red_given_o = p_o_given_red * p_red_box / p_o

# -----------------------------
# 6) PRINT RESULTS
# -----------------------------
print("=== Given information ===")
print(f"P(Box = red)   = {p_red:.2f}")
print(f"P(Box = blue)  = {p_blue:.2f}")
print(f"Red box  : {red_apple_count} apples, {red_orange_count} oranges")
print(f"Blue box : {blue_apple_count} apples, {blue_orange_count} oranges")
print()

print("=== Marginal probabilities of fruit ===")
print(f"P(F = apple)  = {p_a:.4f}  ({p_a*100:.2f}%)")
print(f"P(F = orange) = {p_o:.4f}  ({p_o*100:.2f}%)")
print()

print("=== Posterior probability ===")
print(f"P(Box = red | F = orange) = {p_red_given_o:.4f}  ({p_red_given_o*100:.2f}%)")
