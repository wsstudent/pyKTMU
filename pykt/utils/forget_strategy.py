import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union

class ForgetStrategy:
    """
    Simplified forgetting strategy class focusing on strategy logic.
    
    Supported strategies:
    - random: randomly select users to forget
    - low_performance: select low-performing users to forget
    - high_performance: select high-performing users to forget  
    - low_engagement: select low-engagement users to forget
    - unstable_performance: select users with unstable performance to forget
    """
    
    @staticmethod
    def select_forget_users(df: pd.DataFrame, strategy: str, forget_ratio: float = 0.2, **kwargs) -> Dict[str, List]:
        """
        Select users to forget according to a strategy.
        
        Args:
            df: DataFrame containing user interaction data
            strategy: name of the forgetting strategy
            forget_ratio: forgetting ratio (0-1)
            **kwargs: strategy-specific parameters
            
        Returns:
            Dict containing:
            - 'forget_users': list of user IDs to forget
            - 'retain_users': list of user IDs to retain
            - 'strategy_info': detailed information of the strategy
        """
        
        # Compute per-user statistics
        user_stats = ForgetStrategy._calculate_user_stats(df)
        
        # Strategy dispatcher
        strategy_methods = {
            'random': ForgetStrategy._random_forget,
            'low_performance': ForgetStrategy._low_performance_forget,
            'high_performance': ForgetStrategy._high_performance_forget,
            'low_engagement': ForgetStrategy._low_engagement_forget,
            'unstable_performance': ForgetStrategy._unstable_performance_forget
        }
        
        if strategy not in strategy_methods:
            # Keep original Chinese error message to avoid changing code behavior text
            raise ValueError(f"nsupported strategy: {strategy}")
        
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
        """Compute per-user statistics."""
        # Assume df is an expanded interaction dataset (one interaction per row).
        # If the data is in sequence form, expand it first.
        
        if 'responses' in df.columns:
            # Handle sequence-format data
            expanded_data = []
            for _, row in df.iterrows():
                uid = row['uid']
                responses = [int(x) for x in row['responses'].split(',')]
                for response in responses:
                    expanded_data.append({'uid': uid, 'correct': response})
            expanded_df = pd.DataFrame(expanded_data)
        else:
            # Already expanded
            expanded_df = df.copy()
            if 'user_id' in expanded_df.columns:
                expanded_df['uid'] = expanded_df['user_id']
        
        # Compute metrics
        user_stats = expanded_df.groupby('uid').agg({
            'correct': ['count', 'mean', 'std']
        }).round(4)
        
        user_stats.columns = ['interaction_count', 'accuracy', 'performance_std']
        user_stats['performance_std'] = user_stats['performance_std'].fillna(0)
        
        return user_stats
    
    @staticmethod
    def _random_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """Random forgetting strategy."""
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
        """Low-performance forgetting strategy."""
        # Sort by accuracy ascending and pick the lowest group
        sorted_users = user_stats.sort_values('accuracy')
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
    
    @staticmethod
    def _high_performance_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """High-performance forgetting strategy."""
        # Sort by accuracy descending and pick the highest group
        sorted_users = user_stats.sort_values('accuracy', ascending=False)
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
    
    @staticmethod
    def _low_engagement_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """Low-engagement forgetting strategy."""
        # Sort by interaction count ascending and pick the least engaged users
        sorted_users = user_stats.sort_values('interaction_count')
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
    
    @staticmethod
    def _unstable_performance_forget(user_stats: pd.DataFrame, forget_ratio: float, **kwargs) -> List:
        """Unstable-performance forgetting strategy."""
        # Sort by performance std descending and pick the most unstable users
        sorted_users = user_stats.sort_values('performance_std', ascending=False)
        forget_count = int(len(user_stats) * forget_ratio)
        forget_users = sorted_users.head(forget_count).index.tolist()
        
        return forget_users
