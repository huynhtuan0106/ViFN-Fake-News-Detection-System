"""
DATA BALANCER MODULE
Cân bằng dữ liệu sử dụng SMOTETomek 
- Tỷ lệ target: 60:40 hoặc 65:35 (Real:Fake)
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class SMOTETomekBalancer:
    """
    SMOTETomek balancer cho dữ liệu fake news
    Cân bằng moderate để tránh overfitting
    """
    
    def __init__(self, target_ratio=0.65, random_state=42):
        """
        Args:
            target_ratio: Tỷ lệ class majority sau khi cân bằng (0.6-0.7)
            random_state: Random seed cho reproducibility
        """
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.vectorizer = None
        self.original_distribution = None
        self.balanced_distribution = None
        
    def balance_data(self, df, text_columns=['title_processed', 'summary_processed', 'content_processed']):
        """
        Cân bằng dữ liệu sử dụng SMOTETomek
        
        Args:
            df: DataFrame với text đã processed và label
            text_columns: Các cột text để tạo features
            
        Returns:
            df_balanced: DataFrame đã cân bằng
            balance_info: Thông tin về quá trình cân bằng
        """
        
        print("BALANCING DATA WITH SMOTETOMEK")
        print("="*50)
        
        # 1. Lưu phân phối ban đầu
        self.original_distribution = df['label'].value_counts().sort_index()
        print(f"Original distribution:")
        print(f"   Real (0): {self.original_distribution[0]} ({self.original_distribution[0]/len(df)*100:.1f}%)")
        print(f"   Fake (1): {self.original_distribution[1]} ({self.original_distribution[1]/len(df)*100:.1f}%)")
        
        # 2. Tạo text features cho SMOTE
        print(f"\n🔤 Creating text features for SMOTE...")
        combined_text = self._combine_text_features(df, text_columns)
        
        # 3. Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Giảm features để tránh overfitting
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words=None  # Giữ stop words cho tiếng Việt
        )
        
        X_text = self.vectorizer.fit_transform(combined_text).toarray()
        y = df['label'].values
        
        print(f"   Text features shape: {X_text.shape}")
        
        # 4. Tính toán sampling strategy
        minority_class = 1  # Fake news
        majority_class = 0  # Real news
        
        minority_count = sum(y == minority_class)
        majority_count = sum(y == majority_class)
        
        # Target: Không cân bằng hoàn toàn
        # Ví dụ: Real 65%, Fake 35%
        target_minority_count = int(majority_count * (1 - self.target_ratio) / self.target_ratio)
        target_minority_count = min(target_minority_count, majority_count)  # Không vượt quá majority
        
        # Tạo sampling strategy
        sampling_strategy = {minority_class: target_minority_count}
        
        print(f"\nSampling strategy:")
        print(f"   Target minority class samples: {target_minority_count}")
        print(f"   Will increase from {minority_count} to {target_minority_count}")
        
        # 5. Apply SMOTETomek
        print(f"\nApplying SMOTETomek...")
        
        smote_tomek = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=min(5, minority_count-1)  # Đảm bảo k_neighbors hợp lệ
            ),
            tomek=TomekLinks(sampling_strategy='majority'),
            random_state=self.random_state
        )
        
        try:
            X_balanced, y_balanced = smote_tomek.fit_resample(X_text, y)
            print(f"SMOTETomek completed successfully")
            
        except Exception as e:
            print(f"SMOTETomek failed: {e}")
            print(f"Falling back to SMOTE only...")
            
            # Fallback to SMOTE only
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=min(3, minority_count-1)
            )
            X_balanced, y_balanced = smote.fit_resample(X_text, y)
        
        # 6. Tạo balanced DataFrame
        df_balanced = self._create_balanced_dataframe(df, X_text, X_balanced, y_balanced)
        
        # 7. Thống kê kết quả
        self.balanced_distribution = pd.Series(y_balanced).value_counts().sort_index()
        
        balance_info = self._create_balance_info(df, df_balanced)
        
        print(f"\nBalanced distribution:")
        print(f"   Real (0): {self.balanced_distribution[0]} ({self.balanced_distribution[0]/len(df_balanced)*100:.1f}%)")
        print(f"   Fake (1): {self.balanced_distribution[1]} ({self.balanced_distribution[1]/len(df_balanced)*100:.1f}%)")
        print(f"   Total samples: {len(df)} → {len(df_balanced)}")
        
        return df_balanced, balance_info
    
    def _combine_text_features(self, df, text_columns):
        """Kết hợp các text columns thành một feature"""
        combined_texts = []
        
        for idx, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            
            combined_text = ' '.join(text_parts)
            combined_texts.append(combined_text)
        
        return combined_texts
    
    def _create_balanced_dataframe(self, original_df, X_original, X_balanced, y_balanced):
        """Tạo DataFrame cân bằng từ synthetic samples"""
        
        # Map từ balanced samples về original samples
        balanced_indices = []
        synthetic_rows = []
        
        # Tìm original samples trong balanced set
        for i, x_bal in enumerate(X_balanced):
            # Tìm sample gần nhất trong original data
            distances = np.sum((X_original - x_bal)**2, axis=1)
            closest_idx = np.argmin(distances)
            
            # Nếu distance nhỏ (sample gốc), sử dụng original
            if distances[closest_idx] < 0.01:  # Threshold nhỏ cho original samples
                balanced_indices.append(closest_idx)
            else:
                # Synthetic sample: tạo row mới dựa trên closest original
                synthetic_row = original_df.iloc[closest_idx].copy()
                synthetic_row['label'] = y_balanced[i]
                synthetic_row['is_synthetic'] = True
                synthetic_rows.append(synthetic_row)
        
        # Tạo DataFrame balanced
        if balanced_indices:
            df_original_part = original_df.iloc[balanced_indices].copy()
            df_original_part['is_synthetic'] = False
        else:
            df_original_part = pd.DataFrame()
        
        if synthetic_rows:
            df_synthetic_part = pd.DataFrame(synthetic_rows)
        else:
            df_synthetic_part = pd.DataFrame()
        
        # Kết hợp
        if not df_original_part.empty and not df_synthetic_part.empty:
            df_balanced = pd.concat([df_original_part, df_synthetic_part], ignore_index=True)
        elif not df_original_part.empty:
            df_balanced = df_original_part
        else:
            df_balanced = df_synthetic_part
        
        return df_balanced
    
    def _create_balance_info(self, df_original, df_balanced):
        """Tạo thông tin về quá trình cân bằng"""
        
        balance_info = {
            'original_size': len(df_original),
            'balanced_size': len(df_balanced),
            'original_distribution': self.original_distribution.to_dict(),
            'balanced_distribution': self.balanced_distribution.to_dict(),
            'target_ratio': self.target_ratio,
            'actual_ratio': self.balanced_distribution[0] / len(df_balanced),
            'synthetic_samples': len(df_balanced) - len(df_original),
            'balance_method': 'SMOTETomek',
            'vectorizer_features': self.vectorizer.max_features if self.vectorizer else None
        }
        
        return balance_info

def apply_smotetomek_balancing(df, target_ratio=0.65, random_state=42):
    """
    Hàm tiện ích để apply SMOTETomek balancing
    
    Args:
        df: DataFrame với dữ liệu
        target_ratio: Tỷ lệ majority class mong muốn (0.6-0.7)
        random_state: Random seed
        
    Returns:
        df_balanced: DataFrame đã cân bằng
        balance_info: Thông tin về quá trình cân bằng
    """
    
    balancer = SMOTETomekBalancer(
        target_ratio=target_ratio,
        random_state=random_state
    )
    
    return balancer.balance_data(df)

if __name__ == "__main__":
    print("Testing SMOTETomek Balancer...")
    
    # Tạo sample data
    np.random.seed(42)
    
    # Imbalanced data: 80% real, 20% fake
    n_real = 800
    n_fake = 200
    
    data = {
        'title_processed': (
            ['tin thật ' + str(i) for i in range(n_real)] +
            ['tin giả ' + str(i) for i in range(n_fake)]
        ),
        'summary_processed': (
            ['tóm tắt thật ' + str(i) for i in range(n_real)] +
            ['tóm tắt giả ' + str(i) for i in range(n_fake)]
        ),
        'content_processed': (
            ['nội dung thật rất dài ' + str(i) for i in range(n_real)] +
            ['nội dung giả rất dài ' + str(i) for i in range(n_fake)]
        ),
        'label': [0] * n_real + [1] * n_fake
    }
    
    df_test = pd.DataFrame(data)
    
    # Test balancing
    df_balanced, balance_info = apply_smotetomek_balancing(df_test, target_ratio=0.65)
    
    print(f"\nTest completed!")
    print(f"Balance info: {balance_info}") 