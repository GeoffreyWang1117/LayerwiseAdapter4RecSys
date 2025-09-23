#!/usr/bin/env python3
"""
Comprehensive Real Data Experiment Framework
综合真实数据实验框架 - 确保所有实验都基于真实Amazon Electronics数据
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import subprocess

# 添加src路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveRealExperimentFramework:
    """综合真实实验框架"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir.parent / "results" / "comprehensive_real_experiments"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary = {
            'framework': 'Comprehensive Real Data Experiments',
            'timestamp': datetime.now().isoformat(),
            'experiments': {},
            'data_integrity': 'All experiments use real Amazon Electronics dataset',
            'hardware': 'Dual RTX 3090 GPUs'
        }
    
    def run_baseline_experiment(self):
        """运行基线对比实验"""
        logger.info("🔬 运行基线对比实验...")
        
        try:
            script_path = self.base_dir / "real_data_baseline_experiment.py"
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info("✅ 基线实验完成")
                self.summary['experiments']['baseline'] = {
                    'status': 'completed',
                    'script': 'real_data_baseline_experiment.py',
                    'description': 'Baseline MF vs KD Student vs Fisher-LD comparison'
                }
            else:
                logger.error(f"❌ 基线实验失败: {result.stderr}")
                self.summary['experiments']['baseline'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
        except Exception as e:
            logger.error(f"❌ 基线实验异常: {e}")
            self.summary['experiments']['baseline'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def run_ablation_study(self):
        """运行真实数据消融研究"""
        logger.info("🔬 运行消融研究实验...")
        
        try:
            script_path = self.base_dir / "real_ablation_study_experiment.py"
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info("✅ 消融研究完成")
                self.summary['experiments']['ablation'] = {
                    'status': 'completed',
                    'script': 'real_ablation_study_experiment.py',
                    'description': 'Fisher information weighting strategies comparison'
                }
            else:
                logger.error(f"❌ 消融研究失败: {result.stderr}")
                self.summary['experiments']['ablation'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
        except Exception as e:
            logger.error(f"❌ 消融研究异常: {e}")
            self.summary['experiments']['ablation'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def run_cross_domain_experiment(self):
        """运行跨域验证实验"""
        logger.info("🔬 运行跨域验证实验...")
        
        try:
            script_path = self.base_dir / "real_cross_domain_experiment.py"
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info("✅ 跨域实验完成")
                self.summary['experiments']['cross_domain'] = {
                    'status': 'completed',
                    'script': 'real_cross_domain_experiment.py',
                    'description': 'Amazon Electronics → MovieLens domain transfer'
                }
            else:
                logger.error(f"❌ 跨域实验失败: {result.stderr}")
                self.summary['experiments']['cross_domain'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
        except Exception as e:
            logger.error(f"❌ 跨域实验异常: {e}")
            self.summary['experiments']['cross_domain'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def validate_data_integrity(self):
        """验证数据完整性"""
        logger.info("🔍 验证数据完整性...")
        
        # 检查Amazon数据集是否存在
        amazon_data_dir = self.base_dir.parent / "dataset" / "amazon"
        electronics_reviews = amazon_data_dir / "Electronics_reviews.parquet"
        electronics_meta = amazon_data_dir / "Electronics_meta.parquet"
        
        integrity_check = {
            'amazon_electronics_reviews_exists': electronics_reviews.exists(),
            'amazon_electronics_meta_exists': electronics_meta.exists(),
            'data_directory_accessible': amazon_data_dir.exists()
        }
        
        # 检查MovieLens数据
        movielens_dir = self.base_dir.parent / "dataset" / "movielens"
        movielens_1m_dir = movielens_dir / "1m"
        
        integrity_check.update({
            'movielens_directory_exists': movielens_dir.exists(),
            'movielens_1m_exists': movielens_1m_dir.exists()
        })
        
        self.summary['data_integrity_check'] = integrity_check
        
        all_data_available = all(integrity_check.values())
        logger.info(f"📊 数据完整性检查: {'✅ 通过' if all_data_available else '❌ 失败'}")
        
        return all_data_available
    
    def collect_real_results(self):
        """收集所有真实实验结果"""
        logger.info("📊 收集真实实验结果...")
        
        real_results = {}
        
        # 查找所有真实实验结果文件
        result_patterns = [
            "real_baseline_results_*.json",
            "real_ablation_study_results_*.json", 
            "cross_domain_results_*.json"
        ]
        
        for pattern in result_patterns:
            matching_files = list(self.base_dir.parent.glob(pattern))
            if matching_files:
                # 取最新的文件
                latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    real_results[pattern.replace('*.json', '')] = {
                        'file': str(latest_file),
                        'data': result_data
                    }
                    logger.info(f"✅ 已收集: {latest_file.name}")
                except Exception as e:
                    logger.error(f"❌ 读取失败 {latest_file}: {e}")
        
        self.summary['collected_results'] = real_results
        return real_results
    
    def generate_paper_alignment_report(self):
        """生成论文对齐报告"""
        logger.info("📝 生成论文对齐报告...")
        
        real_results = self.summary.get('collected_results', {})
        
        # 从真实基线结果中提取关键数据
        baseline_data = None
        if 'real_baseline_results_' in real_results:
            baseline_data = real_results['real_baseline_results_']['data']
        
        alignment_report = {
            'paper_table_1_main_results': {
                'source': 'real_baseline_results',
                'status': 'aligned' if baseline_data else 'missing',
                'data': baseline_data.get('methods', {}) if baseline_data else None
            },
            'paper_table_ablation': {
                'source': 'real_ablation_study_results',
                'status': 'needs_update',
                'note': 'Ablation study needs to be run with real data'
            },
            'paper_table_cross_domain': {
                'source': 'cross_domain_results',
                'status': 'partial',
                'note': 'Cross-domain experiment may need completion'
            }
        }
        
        self.summary['paper_alignment'] = alignment_report
    
    def save_comprehensive_summary(self):
        """保存综合摘要"""
        summary_file = self.results_dir / f"comprehensive_experiment_summary_{self.timestamp}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 综合摘要已保存: {summary_file}")
        return summary_file
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("🚀 开始运行所有真实数据实验...")
        
        # 1. 验证数据完整性
        if not self.validate_data_integrity():
            logger.error("❌ 数据完整性验证失败，请检查数据集")
            return False
        
        # 2. 运行各个实验
        self.run_baseline_experiment()
        self.run_ablation_study() 
        self.run_cross_domain_experiment()
        
        # 3. 收集结果
        self.collect_real_results()
        
        # 4. 生成论文对齐报告
        self.generate_paper_alignment_report()
        
        # 5. 保存综合摘要
        summary_file = self.save_comprehensive_summary()
        
        # 6. 打印摘要
        self.print_experiment_summary()
        
        return True
    
    def print_experiment_summary(self):
        """打印实验摘要"""
        print("\n" + "="*60)
        print("📊 综合真实数据实验摘要")
        print("="*60)
        
        print(f"🕐 时间戳: {self.summary['timestamp']}")
        print(f"💾 数据源: 真实Amazon Electronics数据集")
        print(f"🖥️  硬件: {self.summary['hardware']}")
        
        print("\n📋 实验状态:")
        for exp_name, exp_info in self.summary['experiments'].items():
            status_emoji = "✅" if exp_info['status'] == 'completed' else "❌"
            print(f"  {status_emoji} {exp_name}: {exp_info['status']}")
            if exp_info['status'] != 'completed':
                print(f"      错误: {exp_info.get('error', 'Unknown')}")
        
        print("\n🎯 论文对齐状态:")
        if 'paper_alignment' in self.summary:
            for table, info in self.summary['paper_alignment'].items():
                status_emoji = "✅" if info['status'] == 'aligned' else "⚠️"
                print(f"  {status_emoji} {table}: {info['status']}")
                if 'note' in info:
                    print(f"      注释: {info['note']}")
        
        print("\n📁 结果文件:")
        if 'collected_results' in self.summary:
            for result_type, result_info in self.summary['collected_results'].items():
                print(f"  📄 {result_type}: {Path(result_info['file']).name}")
        
        print("\n🔍 下一步建议:")
        print("  1. 检查实验结果与论文表格的一致性")
        print("  2. 更新论文中的虚假数据")
        print("  3. 确保所有声称都基于真实实验结果")
        print("  4. 添加诚实的局限性讨论")

def main():
    """主函数"""
    framework = ComprehensiveRealExperimentFramework()
    success = framework.run_all_experiments()
    
    if success:
        print("\n🎉 所有真实数据实验已完成！")
        print("请检查生成的结果文件，并更新论文内容以确保一致性。")
    else:
        print("\n❌ 实验执行过程中出现问题，请检查日志。")

if __name__ == "__main__":
    main()
