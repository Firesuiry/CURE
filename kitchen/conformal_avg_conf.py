# read all data*.pkl and 计算平均的new_conformal_output中的confidence
import pickle
import glob
import os
import numpy as np

def calculate_average_confidence():
    # 获取所有data*.pkl文件路径
    cache_dir = '/root/autodl-tmp/kitchen/cache'
    pkl_files = glob.glob(os.path.join(cache_dir, 'data*.pkl'))
    
    print(f"找到 {len(pkl_files)} 个pkl文件")
    
    all_confidences = []
    
    for pkl_file in sorted(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 提取new_conformal_output中的confidence
            if 'new_conformal_output' in data:
                new_conformal_output = data['new_conformal_output']
                if isinstance(new_conformal_output, list) and len(new_conformal_output) > 0:
                    # 获取第一个元素的confidence
                    confidence = new_conformal_output[0].get('confidence', None)
                    if confidence is not None:
                        # 如果是numpy类型，转换为float
                        if hasattr(confidence, 'item'):
                            confidence = confidence.item()
                        all_confidences.append(confidence)
                        print(f"{os.path.basename(pkl_file)}: {confidence:.6f}")
                    else:
                        print(f"{os.path.basename(pkl_file)}: 没有找到confidence")
                else:
                    print(f"{os.path.basename(pkl_file)}: new_conformal_output为空或格式不正确")
            else:
                print(f"{os.path.basename(pkl_file)}: 没有找到new_conformal_output")
                
        except Exception as e:
            print(f"读取 {os.path.basename(pkl_file)} 时出错: {e}")
    
    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        print(f"\n总共处理了 {len(all_confidences)} 个有效的confidence值")
        print(f"平均confidence: {avg_confidence:.6f}")
        print(f"最小confidence: {min(all_confidences):.6f}")
        print(f"最大confidence: {max(all_confidences):.6f}")
        print(f"标准差: {np.std(all_confidences):.6f}")
        return avg_confidence
    else:
        print("没有找到任何有效的confidence值")
        return None

if __name__ == "__main__":
    calculate_average_confidence()