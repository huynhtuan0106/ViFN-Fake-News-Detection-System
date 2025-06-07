import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from sklearn.preprocessing import LabelEncoder
# from config import TRUSTED_DOMAINS
import warnings
warnings.filterwarnings('ignore')

class RealDataLoader:
    """
    Class chuyên xử lý dữ liệu thực tế từ file CSV
    Format: domain,title,publish_date,summary,content_html,label
    """
    
    def __init__(self, real_file_path=None, fake_file_path=None):
        self.real_file_path = real_file_path
        self.fake_file_path = fake_file_path
        self.df = None
        self.domain_features = None
        self.domain_dim = None
    
    def load_data(self):
        """Load dữ liệu từ 2 file CSV riêng biệt"""
        
        print("Loading real data from CSV files...")
        
        # Load real news data
        if self.real_file_path:
            try:
                df_real = pd.read_csv(self.real_file_path, encoding='utf-8')
                # Đảm bảo label = 0 cho tin thật
                df_real['label'] = 0  
                print(f"Loaded {len(df_real)} real news articles")
                print(f"Real data columns: {df_real.columns.tolist()}")
            except Exception as e:
                print(f"Error loading real data: {e}")
                df_real = pd.DataFrame()
        else:
            print("No real data file provided")
            df_real = pd.DataFrame()
        
        # Load fake news data  
        if self.fake_file_path:
            try:
                df_fake = pd.read_csv(self.fake_file_path, encoding='utf-8')
                # Đảm bảo label = 1 cho tin giả
                df_fake['label'] = 1  
                print(f"Loaded {len(df_fake)} fake news articles")
                print(f"Fake data columns: {df_fake.columns.tolist()}")
            except Exception as e:
                print(f"Error loading fake data: {e}")
                df_fake = pd.DataFrame()
        else:
            print("No fake data file provided")
            df_fake = pd.DataFrame()
        
        # Combine datasets
        if not df_real.empty and not df_fake.empty:
            self.df = pd.concat([df_real, df_fake], ignore_index=True)
        elif not df_real.empty:
            self.df = df_real
        elif not df_fake.empty:
            self.df = df_fake
        else:
            raise ValueError("No valid data files provided or files are empty")
        
        print(f"Total dataset size: {len(self.df)}")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
        
        return self.df
    
    def clean_and_validate_data(self):
        """Làm sạch và validate dữ liệu"""
        
        print("\nCleaning and validating data...")
        
        # Kiểm tra các cột cần thiết theo format thực tế
        required_columns = ['domain', 'title']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Available columns: {self.df.columns.tolist()}")
        
        # Xử lý content_html -> content
        if 'content_html' in self.df.columns:
            print("Found content_html column, processing HTML content...")
            self.df['content'] = self.df['content_html'].apply(self._clean_html_content)
        elif 'content' not in self.df.columns:
            print("Warning: No content column found, creating empty content")
            self.df['content'] = ''
        
        # Xử lý summary
        if 'summary' not in self.df.columns:
            print("Warning: No summary column found, creating from title")
            self.df['summary'] = self.df['title']
        
        # Xử lý missing values cho text columns
        text_columns = ['title', 'summary', 'content']
        for col in text_columns:
            if col in self.df.columns:
                original_nulls = self.df[col].isnull().sum()
                self.df[col] = self.df[col].fillna('')
                self.df[col] = self.df[col].astype(str)
                if original_nulls > 0:
                    print(f"Filled {original_nulls} null values in {col}")
        
        # Xử lý domain
        domain_nulls = self.df['domain'].isnull().sum()
        self.df['domain'] = self.df['domain'].fillna('unknown')
        self.df['domain'] = self.df['domain'].astype(str)
        if domain_nulls > 0:
            print(f"Filled {domain_nulls} null values in domain")
        
        # Loại bỏ các bài viết quá ngắn
        min_content_length = 50  # Tối thiểu 50 ký tự
        before_filter = len(self.df)
        
        # Tính tổng độ dài content
        self.df['total_content_length'] = (
            self.df['title'].str.len() + 
            self.df['summary'].str.len() + 
            self.df['content'].str.len()
        )
        
        self.df = self.df[self.df['total_content_length'] >= min_content_length]
        after_filter = len(self.df)
        
        print(f"Filtered out {before_filter - after_filter} articles with insufficient content")
        print(f"Remaining articles: {after_filter}")
        
        # Kiểm tra balance của labels
        label_counts = self.df['label'].value_counts()
        print(f"Final label distribution:")
        print(f"  Real news (0): {label_counts.get(0, 0)}")
        print(f"  Fake news (1): {label_counts.get(1, 0)}")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        return self.df
    
    def _clean_html_content(self, html_content):
        """Làm sạch nội dung HTML"""
        if pd.isna(html_content) or html_content == '':
            return ''
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Loại bỏ script và style
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Lấy text
            text = soup.get_text()
            
            # Làm sạch whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return str(html_content)
    
    def preprocess_text(self):
        """Tiền xử lý văn bản tiếng Việt"""
        
        print("\nPreprocessing Vietnamese text...")
        
        text_columns = ['title', 'summary', 'content']
        
        for col in text_columns:
            if col in self.df.columns:
                print(f"Processing {col}...")
                self.df[f'{col}_processed'] = self.df[col].apply(self._preprocess_vietnamese_text)
                
                # Thống kê
                avg_length_before = self.df[col].str.len().mean()
                avg_length_after = self.df[f'{col}_processed'].str.len().mean()
                print(f"  Average length before: {avg_length_before:.1f}")
                print(f"  Average length after: {avg_length_after:.1f}")
        
        print("Text preprocessing completed!")
        return self.df
    
    def _preprocess_vietnamese_text(self, text):
        """Xử lý văn bản tiếng Việt"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Loại bỏ email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Giữ lại chữ cái, số và ký tự tiếng Việt
        vietnamese_chars = r'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        pattern = f'[^\\w\\s{vietnamese_chars}]'
        text = re.sub(pattern, ' ', text)
        
        # Loại bỏ số dư thừa
        text = re.sub(r'\d+', ' ', text)
        
        # Loại bỏ whitespace thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize tiếng Việt
        try:
            text = ViTokenizer.tokenize(text)
        except:
            pass
        
        return text
    
    def process_domains(self):
        """Xử lý domain features"""
        
        print("\nProcessing domain features...")
        
        # Tính credibility score
        self.df['domain_credibility'] = self.df['domain'].apply(self._domain_credibility_score)
        
        # One-hot encode domains
        domain_encoder = LabelEncoder()
        domain_encoded = domain_encoder.fit_transform(self.df['domain'])
        domain_onehot = np.eye(len(domain_encoder.classes_))[domain_encoded]
        
        # Kết hợp features
        self.domain_features = np.column_stack([
            domain_onehot, 
            self.df['domain_credibility'].values.reshape(-1, 1)
        ])
        
        self.domain_dim = self.domain_features.shape[1]
        
        print(f"Domain features shape: {self.domain_features.shape}")
        print(f"Unique domains: {len(domain_encoder.classes_)}")
        print(f"Domain distribution:")
        print(self.df['domain'].value_counts().head(10))
        
        return self.domain_features, self.domain_dim
    
    def _domain_credibility_score(self, domain):
        """Tính điểm tin cậy cho domain"""
        domain = str(domain).lower()
        
        # if domain in [d.lower() for d in TRUSTED_DOMAINS]:
        #     return 1.0
        if domain.endswith('.gov.vn') or domain.endswith('.edu.vn'):
            return 0.5
        elif domain.endswith('.vn'):
            return 0.4
        elif domain.endswith('.com') or domain.endswith('.net') or domain.endswith('.org'):
            return 0.2
        else:
            return 0.1
    
    def get_statistics(self):
        """Lấy thống kê dữ liệu"""
        
        stats = {
            'total_articles': len(self.df),
            'real_articles': len(self.df[self.df['label'] == 0]),
            'fake_articles': len(self.df[self.df['label'] == 1]),
            'unique_domains': self.df['domain'].nunique(),
            'avg_title_length': self.df['title'].str.len().mean(),
            'avg_content_length': self.df['content'].str.len().mean(),
        }
        
        if 'summary' in self.df.columns:
            stats['avg_summary_length'] = self.df['summary'].str.len().mean()
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\nTop domains:")
        print(self.df['domain'].value_counts().head(10))
        
        return stats
    
    def save_processed_data(self, output_path='processed_data.csv'):
        """Lưu dữ liệu đã xử lý"""
        
        self.df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nProcessed data saved to: {output_path}")
        
        return output_path

def load_and_preprocess_real_data(real_file_path, fake_file_path):
    """
    Function chính để load và xử lý dữ liệu thực tế
    
    Args:
        real_file_path: Đường dẫn file CSV chứa tin thật
        fake_file_path: Đường dẫn file CSV chứa tin giả
    
    Returns:
        df: DataFrame đã xử lý
        domain_features: Features của domain
        domain_dim: Số chiều domain features
    """
    
    # Khởi tạo data loader
    loader = RealDataLoader(real_file_path, fake_file_path)
    
    # Load dữ liệu
    df = loader.load_data()
    
    # Làm sạch và validate
    df = loader.clean_and_validate_data()
    
    # Tiền xử lý text
    df = loader.preprocess_text()
    
    # Xử lý domain
    domain_features, domain_dim = loader.process_domains()
    
    # Lấy thống kê
    stats = loader.get_statistics()
    
    return df, domain_features, domain_dim

if __name__ == "__main__":
    # Test với dữ liệu thực tế
    real_file = "articles/real.csv"
    fake_file = "articles/fake.csv"
    
    print("Testing data loader with real data...")
    print(f"Real file: {real_file}")
    print(f"Fake file: {fake_file}")
    
    try:
        df, domain_features, domain_dim = load_and_preprocess_real_data(
            real_file, fake_file
        )
        print("\nData loading and preprocessing completed successfully!")
        print(f"Final dataset shape: {df.shape}")
        print(f"Domain features shape: {domain_features.shape}")
        
    except Exception as e:
        print(f"Error in data processing: {e}")
        print("Please check your file paths and data format.") 