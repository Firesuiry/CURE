import pickle
class ListDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super().__getitem__(key)
with open('/root/autodl-tmp/kitchen/pickle/result_with_confidence.pkl', 'rb') as f:
    result_with_confidence = pickle.load(f)

rwc = result_with_confidence['model0']

# 将rwc随机分为两个list
import random
random.shuffle(rwc)
rwc1 = rwc[:len(rwc)//2]
rwc2 = rwc[len(rwc)//2:]

# 校准算法实现
def calculate_success_rate_with_mark(data, confidence_mark):
    """
    计算给定confidence_mark下的成功率
    当confidence < mark时，标记为成功
    当confidence >= mark时，保持原有的成功/失败状态
    """
    total = len(data)
    if total == 0:
        return 0.0
    
    success_count = 0
    for confidence, original_success in data:
        if confidence < confidence_mark:
            # confidence小于mark时，标记为成功
            success_count += 1
        else:
            # confidence大于等于mark时，保持原有状态
            if original_success:
                success_count += 1
    
    return success_count / total

# 获取所有可能的confidence值作为候选mark
all_confidences = [item[0] for item in rwc1]
confidence_marks = sorted(set(all_confidences))

# 存储差值的列表
differences = []

# 存储所有结果用于查找最接近0.85的成功率
results = []

# 对每个confidence_mark进行校准和测试
for mark in confidence_marks:
    # 在rwc1上计算成功率（校准）
    success_rate_rwc1 = calculate_success_rate_with_mark(rwc1, mark)
    
    # 在rwc2上测试相同mark的成功率
    success_rate_rwc2 = calculate_success_rate_with_mark(rwc2, mark)
    
    # 计算差值
    difference = abs(success_rate_rwc1 - success_rate_rwc2)
    differences.append(difference)
    
    # 存储结果
    results.append({
        'mark': mark,
        'rwc1_success_rate': success_rate_rwc1,
        'rwc2_success_rate': success_rate_rwc2,
        'difference': difference
    })
    
    print(f"Mark: {mark:.4f}, RWC1成功率: {success_rate_rwc1:.4f}, RWC2成功率: {success_rate_rwc2:.4f}, 差值: {difference:.4f}")

# 计算平均差值
average_difference = sum(differences) / len(differences) if differences else 0
print(f"\n平均差值: {average_difference:.4f}")
print(f"总共测试了 {len(confidence_marks)} 个confidence mark")

# 找到成功率最接近0.85的mark
target_success_rate = 0.85
best_match = None
min_distance = float('inf')

for result in results:
    # 使用rwc1的成功率来寻找最接近0.85的mark
    distance = abs(result['rwc1_success_rate'] - target_success_rate)
    if distance < min_distance:
        min_distance = distance
        best_match = result

if best_match:
    print(f"\n=== 最接近成功率0.85的结果 ===")
    print(f"Confidence Mark: {best_match['mark']:.4f}")
    print(f"RWC1成功率: {best_match['rwc1_success_rate']:.4f}")
    print(f"RWC2成功率: {best_match['rwc2_success_rate']:.4f}")
    print(f"与目标0.85的差距: {min_distance:.4f}")
    print(f"此时的差值: {best_match['difference']:.4f}")
else:
    print("\n未找到合适的结果")



