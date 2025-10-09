# model0

import pickle
class ListDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super().__getitem__(key)
with open('/root/autodl-tmp/kitchen/pickle/result_with_confidence.pkl', 'rb') as f:
    result_with_confidence = pickle.load(f)

def calculate_overstep_rate(rwc, target_success_rate=0.9):
    """
    计算overstep rate
    1. 找到使成功率达到target_success_rate的confidence mark
    2. 统计confidence大于mark且success为false的比例
    
    Args:
        rwc: list of tuples (confidence, success)
        target_success_rate: 目标成功率，默认0.85
    
    Returns:
        tuple: (confidence_mark, overstep_rate)
    """
    if not rwc:
        return None, 0.0
    
    # 获取所有可能的confidence值作为候选mark
    all_confidences = [item[0] for item in rwc]
    confidence_marks = sorted(set(all_confidences))
    
    # 找到最接近目标成功率的confidence mark
    best_mark = None
    min_distance = float('inf')
    
    for mark in confidence_marks:
        # 计算在该mark下的成功率
        total = len(rwc)
        success_count = 0
        
        for confidence, original_success in rwc:
            if confidence < mark:
                # confidence小于mark时，标记为成功
                success_count += 1
            else:
                # confidence大于等于mark时，保持原有状态
                if original_success:
                    success_count += 1
        
        success_rate = success_count / total
        distance = abs(success_rate - target_success_rate)
        
        if distance < min_distance:
            min_distance = distance
            best_mark = mark
    
    if best_mark is None:
        return None, 0.0
    
    # 计算overstep rate：confidence大于mark且success为false的比例
    high_confidence_tasks = [item for item in rwc if item[0] >= best_mark]
    
    if not high_confidence_tasks:
        return best_mark, 0.0
    
    failed_high_confidence = [item for item in high_confidence_tasks if not item[1]]
    overstep_rate = len(failed_high_confidence) / len(high_confidence_tasks)
    
    return best_mark, overstep_rate

def calculate_top_confidence_failure_rate(rwc, top_percentage=0.35):
    """
    计算置信度最高的X%任务中失败任务的比例
    
    Args:
        rwc: list of tuples (confidence, success)
        top_percentage: 要考虑的高置信度任务的百分比，默认0.35（35%）
    
    Returns:
        float: 高置信度任务中的失败率
    """
    if not rwc:
        return 0.0
    
    # 按置信度降序排序
    sorted_tasks = sorted(rwc, key=lambda x: x[0], reverse=True)
    
    # 计算前X%的任务数量
    total_tasks = len(sorted_tasks)
    top_count = int(total_tasks * top_percentage)
    
    if top_count == 0:
        return 0.0
    
    # 获取前X%的任务
    top_tasks = sorted_tasks[:top_count]
    
    # 计算这些任务中的失败率
    failures = sum(1 for task in top_tasks if not task[1])
    failure_rate = failures / len(top_tasks)
    
    return failure_rate

def calculate_help_seeking_rate(rwc, confidence_mark):
    """
    计算求助率：confidence低于mark所占的比例
    
    Args:
        rwc: List of tuples (confidence, success)
        confidence_mark: 置信度阈值
    
    Returns:
        float: 求助率
    """
    if not rwc:
        return 0.0
    
    total_tasks = len(rwc)
    help_seeking_tasks = sum(1 for task in rwc if task[0] < confidence_mark)
    
    return help_seeking_tasks / total_tasks

def calculate_over_inquiry_rate(rwc, confidence_mark):
    """
    计算过度询问率：confidence低于mark中success为True的比例
    
    Args:
        rwc: List of tuples (confidence, success)
        confidence_mark: 置信度阈值
    
    Returns:
        float: 过度询问率
    """
    if not rwc:
        return 0.0
    
    # 找到所有confidence低于mark的任务
    low_confidence_tasks = [task for task in rwc if task[0] < confidence_mark]
    
    if not low_confidence_tasks:
        return 0.0
    
    # 计算这些任务中success为True的比例
    successful_low_confidence = sum(1 for task in low_confidence_tasks if task[1])
    
    return successful_low_confidence / len(low_confidence_tasks)

# 加载三个模型的数据
rwc_model0 = result_with_confidence['model0']

# introplan+CURE
with open('/root/autodl-tmp/kitchen/pickle/introplan_result_with_confidence.pkl', 'rb') as f:
    intro_plan_cure_rwc = pickle.load(f)

# introplan
with open('/root/autodl-tmp/kitchen/introplan/introplan_result_with_confidence.pkl', 'rb') as f:
    intro_plan_rwc = pickle.load(f)

# 对三个模型分别计算overstep rate
print("=== Overstep Rate 分析结果 ===")
print()

# Model 0
mark0, overstep0 = calculate_overstep_rate(rwc_model0)
print(f"Model 0:")
print(f"  Confidence Mark (85%成功率): {mark0:.4f}")
print(f"  Overstep Rate: {overstep0:.4f} ({overstep0*100:.2f}%)")
print()

# Introplan + CURE
mark1, overstep1 = calculate_overstep_rate(intro_plan_cure_rwc)
print(f"Introplan + CURE:")
print(f"  Confidence Mark (85%成功率): {mark1:.4f}")
print(f"  Overstep Rate: {overstep1:.4f} ({overstep1*100:.2f}%)")
print()

# Introplan
mark2, overstep2 = calculate_overstep_rate(intro_plan_rwc)
print(f"Introplan:")
print(f"  Confidence Mark (85%成功率): {mark2:.4f}")
print(f"  Overstep Rate: {overstep2:.4f} ({overstep2*100:.2f}%)")
print()

# 总结
print("=== 总结 ===")
print(f"Model 0 overstep rate: {overstep0*100:.2f}%")
print(f"Introplan + CURE overstep rate: {overstep1*100:.2f}%")
print(f"Introplan overstep rate: {overstep2*100:.2f}%")

# print("\n=== 置信度最高35%中的失败率 ===")
# # 计算三个模型中置信度最高35%任务的失败率
# failure_rate0 = calculate_top_confidence_failure_rate(rwc_model0)
# failure_rate1 = calculate_top_confidence_failure_rate(intro_plan_cure_rwc)
# failure_rate2 = calculate_top_confidence_failure_rate(intro_plan_rwc)

# print(f"Model 0 - 置信度最高35%中失败率: {failure_rate0*100:.2f}%")
# print(f"Introplan + CURE - 置信度最高35%中失败率: {failure_rate1*100:.2f}%")
# print(f"Introplan - 置信度最高35%中失败率: {failure_rate2*100:.2f}%")

print("\n=== 求助率分析 ===\n")
# 计算三个模型的求助率
help_seeking_rate0 = calculate_help_seeking_rate(rwc_model0, mark0)
help_seeking_rate1 = calculate_help_seeking_rate(intro_plan_cure_rwc, mark1)
help_seeking_rate2 = calculate_help_seeking_rate(intro_plan_rwc, mark2)

print(f"Model 0 - 求助率: {help_seeking_rate0*100:.2f}%")
print(f"Introplan + CURE - 求助率: {help_seeking_rate1*100:.2f}%")
print(f"Introplan - 求助率: {help_seeking_rate2*100:.2f}%")

print("\n=== 过度询问率分析 ===\n")
# 计算三个模型的过度询问率
over_inquiry_rate0 = calculate_over_inquiry_rate(rwc_model0, mark0)
over_inquiry_rate1 = calculate_over_inquiry_rate(intro_plan_cure_rwc, mark1)
over_inquiry_rate2 = calculate_over_inquiry_rate(intro_plan_rwc, mark2)

print(f"Model 0 - 过度询问率: {over_inquiry_rate0*100:.2f}%")
print(f"Introplan + CURE - 过度询问率: {over_inquiry_rate1*100:.2f}%")
print(f"Introplan - 过度询问率: {over_inquiry_rate2*100:.2f}%")

def calculate_sr_hr_area(rwc):
    """
    计算SR-HR Area指标，参考result_generate.py的逻辑
    
    Args:
        rwc: List of tuples (confidence, success)
    
    Returns:
        float: sr_hr_area值
    """
    import numpy as np
    import copy
    
    # 按置信度排序
    sorted_data = sorted(rwc, key=lambda x: x[0])
    
    # 计算不同Help Rate下的Success Rate
    success_rates = []
    for hr_percent in range(0, 101, 1):
        success_cache = []
        for ii, (conf, success) in enumerate(sorted_data):
            if ii < len(sorted_data) * hr_percent / 100:
                success_cache.append(1)
            elif success:
                success_cache.append(1)
            else:
                success_cache.append(0)
        
        if len(success_cache) > 0:
            success_rates.append(np.mean(success_cache))
        else:
            success_rates.append(1)
    
    # 均一化处理
    success_rates = np.array(success_rates)
    data_copy = copy.deepcopy(success_rates)
    
    # 减去线性基准线
    minus_data = np.linspace(success_rates[0], 1, len(success_rates))
    data_copy -= minus_data
    
    # 归一化系数
    normal_coff = (1 - success_rates[0]) * success_rates[0]
    data_copy /= normal_coff
    
    # 计算面积
    divider = 0.5 * (len(success_rates) - 1)
    sr_hr_area = np.trapz(data_copy, dx=1) / divider
    
    return sr_hr_area

print("\n=== SR-HR Area 分析 ===\n")
# 计算三个模型的SR-HR Area
sr_hr_area0 = calculate_sr_hr_area(rwc_model0)
sr_hr_area1 = calculate_sr_hr_area(intro_plan_cure_rwc)
sr_hr_area2 = calculate_sr_hr_area(intro_plan_rwc)

print(f"Model 0 - SR-HR Area: {sr_hr_area0:.6f}")
print(f"Introplan + CURE - SR-HR Area: {sr_hr_area1:.6f}")
print(f"Introplan - SR-HR Area: {sr_hr_area2:.6f}")

