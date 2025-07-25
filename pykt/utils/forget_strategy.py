import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union

class ForgetStrategy:
    """
    简化的遗忘策略类，专注于策略逻辑
    
    支持的策略：
    - random: 随机选择用户进行遗忘
    - low_performance: 选择低表现用户进行遗忘
    - high_performance: 选择高表现用户进行遗忘  
    - low_engagement: 选择低参与度用户进行遗忘
    - unstable_performance: 选择表现不稳定的用户进行遗忘
    """
    
    @staticmethod
    def select_forget_users(df: pd.DataFrame, strategy: str, forget_ratio: float = 0.2, **kwargs) -> Dict[str, List]:
        """
        根据策略选择需要遗忘的用户
        
        Args:
            df: 包含用户交互数据的DataFrame
            strategy: 遗忘策略名称
            forget_ratio: 遗忘比例 (0-1)
            **kwargs: 策略特定参数
            
        Returns:
            Dict包含:
            - 'forget_users': 需要遗忘的用户ID列表
            - 'retain_users': 需要保留的用户ID列表
            - 'strategy_info': 策略详细信息
        """
        
        # 计算用户统计信息
        user_stats = ForgetStrategy._calculate_user_stats(df)
        
        # 根据策略选择用户
        strategy_methods = {
            'random': ForgetStrategy._random_forget,
            'low_performance': ForgetStrategy._low_performance_forget,
            'high_performance': ForgetStrategy._high_performance_forget,
            'low_engagement': ForgetStrategy._low_engagement_forget,
            'unstable_performance': ForgetStrategy._unstable_performance_forget
        }
        
        if strategy not in strategy_methods:
            raise ValueError(f"不支持的策略: {strategy}")
        
        forget_users = strategy_methods[strategy](user_stats, forget_ratio, **kwargs)
        all_users = user_stats.index.tolist()
        retain_users = [u for u in all_users if u not in forget_users]
        
        return {
            'forget_users': forget_users,
            'retain_users': retain_users,
            'strategy_info': {
                'strategy': strategy,
                'forget_ratio': forget_ratio,
                'total_users': len(all_users),
                'forget_count': len(forget_users),
                'retain_count': len(retain_users),
                'actual_forget_ratio': len(forget_users) / len(all_users)
            }
        }
    
    @staticmethod
    def _calculate_user_stats(df: pd.DataFrame) -> pd.DataFrame:
        """计算用户统计信息"""
        # 假设df已经是展开的交互数据，每行代表一次交互
        # 如果是序列格式，需要先展开
        
        if 'responses' in df.columns:
            # 处理序列格式数据
            expanded_data = []
            for _, row in df.iterrows():
                uid = row['uid']
                responses = [int(x) for x in row['responses'].split(',')]
                for response in responses:
                    expanded_data.append({'uid': uid, 'correct': response})
            expanded_df = pd.DataFrame(expanded_data)
        else:
            # 已经是展开格式
            expanded_df = df.copy()
            if 'user_id' in expanded_df.columns:
                expanded_df['uid'] = expanded_df['user_id']
        
        # 计算统计指标
        user_stats = expanded_df.groupby('uid').agg({
            'correct': ['count', 'mean', 'std']
        }).round(4)
        
        user_stats.columns = ['interaction_count', 'accuracy', 'performance_std']
        user_stats['performance_std'] = user_stats['performance_std'].fillna(0)
        
        return user_stats
    
    @staticmethod
    def _random_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """随机遗忘策略"""
        np.random.seed(kwargs.get('seed', 42))
        total_users = len(user_stats)
        forget_count = int(total_users * forget_ratio)
        
        forget_users = np.random.choice(
            user_stats.index.tolist(), 
            size=forget_count, 
            replace=False
        ).tolist()
        
        return forget_users
    
    @staticmethod
    def _low_performance_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """低表现遗忘策略"""
        # 按准确率排序，选择最低的用户
        sorted_users = user_stats.sort_values('accuracy')
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
    
    @staticmethod
    def _high_performance_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """高表现遗忘策略"""
        # 按准确率排序，选择最高的用户
        sorted_users = user_stats.sort_values('accuracy', ascending=False)
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
    
    @staticmethod
    def _low_engagement_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """低参与度遗忘策略"""
        # 按交互次数排序，选择最少的用户
        sorted_users = user_stats.sort_values('interaction_count')
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
    
    @staticmethod
    def _unstable_performance_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """不稳定表现遗忘策略"""
        # 按表现标准差排序，选择最不稳定的用户
        sorted_users = user_stats.sort_values('performance_std', ascending=False)
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
